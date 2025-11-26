import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import os
import shutil

# ==========================================
# 1. TRAJECTORY RECONSTRUCTION ENGINE
# ==========================================

def get_aircraft_trajectory_points(row, x_faf, y_faf, r):
    """
    Returns the 4 key waypoints and segment durations for an aircraft:
    1. Entry
    2. Turn Start
    3. Turn End
    4. FAF
    """
    # 1. Inputs
    d_i = row['d_i']
    entry_x, entry_y = row['x_entry'], row['y_entry']
    is_north = row['is_north']
    is_long = row['long_arc']
    
    # Speeds (nm/s)
    vL_sec = row['v_L'] / 3600.0
    vT_sec = row['v_theta'] / 3600.0
    vF_sec = row['v_f'] / 3600.0
    
    # 2. Geometry Centers
    y_off = r if is_north else -r
    Cx, Cy = x_faf - d_i, y_faf + y_off
    
    # 3. Turn End Point (Tangent to Final Leg)
    # The turn always ends exactly at the d_i extension point on the X-axis (offset by center Y)
    # Actually, based on logic: Center is at (Cx, Cy). Final leg is horizontal at y=y_faf.
    # So the circle is tangent to the final leg at (Cx, y_faf).
    end_x, end_y = Cx, y_faf 
    
    # 4. Turn Start Point (Tangent to Entry Leg)
    # Vector from Center to Entry
    dx, dy = entry_x - Cx, entry_y - Cy
    d0 = np.sqrt(dx**2 + dy**2)
    
    # Angle of Entry relative to Center
    alpha = np.arctan2(dy, dx)
    
    # Angle offset to tangent point (acos(r/d0))
    # Direction depends on CW vs CCW. 
    # North Center (CCW flow): Subtract offset? 
    # South Center (CW flow): Add offset?
    beta = np.arccos(r / d0)
    
    # Determination of turn direction based on geometry:
    # If North (Center Above): We arrive from Top-Right or Top-Left. 
    # We turn 'around' the bottom of the circle. This is CCW.
    # Tangent point angle = alpha - beta
    if is_north:
        theta_start = alpha - beta
    else:
        theta_start = alpha + beta
        
    start_x = Cx + r * np.cos(theta_start)
    start_y = Cy + r * np.sin(theta_start)
    
    # 5. Arc Length & Duration
    # Angle at End: North Center -> 270 deg (-pi/2). South Center -> 90 deg (pi/2)
    theta_end = -np.pi/2 if is_north else np.pi/2
    
    # Normalize angles to 0..2pi for math
    ts_norm = theta_start % (2*np.pi)
    te_norm = theta_end % (2*np.pi)
    
    if is_north: # CCW Turn
        if is_long:
            # Long arc CCW
            # If current < target, we just go forward. If current > target, we wrap.
            # Actually, calculate diff
            diff = (te_norm - ts_norm) % (2*np.pi)
            # Long arc implies we took the "other" tangent? 
            # Simplified: Use the theta_rad calculated in Pyomo to calculate distance
            pass 
        else:
            pass
            
    # --- SIMPLIFIED RECONSTRUCTION USING LENGTHS ---
    # Because angle logic is tricky with Long/Short flags, we use the EXACT lengths 
    # computed by the optimizer to interpolate, ensuring consistency.
    
    # Recalculate Lengths from Pyomo logic (guaranteed correct)
    d0_sq = (entry_x - Cx)**2 + (entry_y - Cy)**2
    d0_p_sq = (entry_x - (x_faf-d_i))**2 + (entry_y - y_faf)**2
    d_L_len = np.sqrt(d0_sq - r**2)
    
    # Recompute Angle Sweep (rad) exactly as Pyomo did
    d0_val = np.sqrt(d0_sq)
    term1 = np.clip(r/d0_val, -1, 1)
    term2 = np.clip((r**2 + d0_sq - d0_p_sq)/(2*r*d0_val), -1, 1)
    t1 = np.arccos(term1)
    t2 = np.arccos(term2)
    theta_rad = (2*np.pi - (t1+t2)) if is_long else (t2-t1)
    d_theta_len = r * theta_rad
    d_final_len = d_i
    
    # Durations
    t_L = d_L_len / vL_sec
    t_turn = d_theta_len / vT_sec
    t_final = d_final_len / vF_sec
    
    return {
        'pts': [(entry_x, entry_y), (start_x, start_y), (end_x, end_y), (x_faf, y_faf)],
        'lens': [d_L_len, d_theta_len, d_final_len],
        'times': [row['entry_time'], row['entry_time']+t_L, row['entry_time']+t_L+t_turn, row['arrival_time']],
        'angles': (theta_start, theta_end, is_north, theta_rad), # For arc drawing
        'center': (Cx, Cy)
    }

def interpolate_position(info, current_time):
    """
    Given the trajectory info and current time, return (x, y).
    Returns None if aircraft is not in the system.
    """
    times = info['times']
    
    # Not entered yet
    if current_time < times[0]: return None
    # Already landed (Keep at FAF or hide? Let's hide 10s after landing)
    if current_time > times[3]: 
        if current_time > times[3] + 10: return None
        return info['pts'][3] # Stuck at FAF
    
    pts = info['pts']
    
    # Segment 1: Tangent (Straight)
    if current_time <= times[1]:
        pct = (current_time - times[0]) / (times[1] - times[0])
        x = pts[0][0] + pct * (pts[1][0] - pts[0][0])
        y = pts[0][1] + pct * (pts[1][1] - pts[0][1])
        return x, y
        
    # Segment 2: Turn (Circular)
    if current_time <= times[2]:
        pct = (current_time - times[1]) / (times[2] - times[1])
        
        # Unpack Arc Info
        start_ang, end_ang_ref, is_north, sweep_rad = info['angles']
        cx, cy = info['center']
        r = R_TURN_NM
        
        # Calculate Current Angle
        # Direction: North Centers turn CCW (+), South Centers turn CW (-)
        # Note: This direction logic depends on the exact vector math.
        # Heuristic: We simply interpolate from Start Angle by 'sweep_rad'
        # We need to know if we add or subtract sweep_rad.
        # North Center (CCW): Angle increases? South (CW): Decreases?
        
        direction = 1 if is_north else -1
        # Check Long Arc logic? 
        # Actually, simpler: We simply interp from Start Angle towards End Angle?
        # No, because of the 0/360 wrap, linear interp of angles is risky.
        # Robust: Use the sweep direction derived from geometry.
        
        # Let's assume standard flow: 
        # North Arr -> Left Turn (CCW) -> direction = 1
        # South Arr -> Right Turn (CW) -> direction = -1
        # Correction: If Long Arc, we go the 'long way'.
        
        # Let's try simple interpolation of the vector
        # It's an arc, so we need angles.
        
        # Correct logic using start/end check:
        current_angle = start_ang + direction * (pct * sweep_rad)
        
        # Re-check: If long arc, does direction flip? 
        # No, "Long Arc" just means sweep_rad is large (> pi). 
        # The direction of turn (Left vs Right) is determined by North/South entry.
        
        x = cx + r * np.cos(current_angle)
        y = cy + r * np.sin(current_angle)
        return x, y
        
    # Segment 3: Final (Straight)
    if current_time <= times[3]:
        pct = (current_time - times[2]) / (times[3] - times[2])
        x = pts[2][0] + pct * (pts[3][0] - pts[2][0])
        y = pts[2][1] + pct * (pts[3][1] - pts[2][1])
        return x, y
        
    return None

# ==========================================
# 2. GENERATION LOOP
# ==========================================

def generate_scenario_gif(df_res, fps=5, speed_up=5):
    """
    fps: Frames per second of GIF
    speed_up: Simulation speed multiplier (5x means 1 sec GIF = 5 sec Sim)
    """
    
    print("Pre-computing trajectories...")
    traj_db = {}
    r = R_TURN_NM
    x_faf, y_faf = processed_fixes['VINII']['x'], processed_fixes['VINII']['y']
    
    # 1. Precompute
    for idx, row in df_res.iterrows():
        traj_db[row['entry_id']] = get_aircraft_trajectory_points(row, x_faf, y_faf, r)
        
    # 2. Setup Time
    t_start = df_res['entry_time'].min()
    t_end = df_res['arrival_time'].max() + 20
    dt = speed_up # Time step in simulation seconds
    
    sim_times = np.arange(t_start, t_end, dt)
    print(f"Generating {len(sim_times)} frames...")
    
    # 3. Setup Plot
    filenames = []
    output_folder = 'sim_frames'
    if os.path.exists(output_folder): shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    # Bounds
    all_x = df_res['x_entry'].tolist() + [x_faf]
    all_y = df_res['y_entry'].tolist() + [y_faf]
    pad = 5
    x_lim = (min(all_x)-pad, max(all_x)+pad)
    y_lim = (min(all_y)-pad, max(all_y)+pad)
    
    # Colors
    colors = {'NorthWest': 'red', 'NorthEast': 'blue', 'SouthEast': 'green', 'SouthWest': 'orange'}
    
    for i, t in enumerate(sim_times):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=80)
        
        # Background: FAF and Runway
        ax.plot(x_faf, y_faf, 'k^', markersize=10)
        ax.plot(0, 0, 'k*', markersize=12)
        ax.text(x_faf, y_faf+1, "FAF", ha='center')
        ax.text(0, 0+1, "RWY", ha='center')
        
        # Draw full paths (faintly) for context
        for eid, info in traj_db.items():
            # Extract waypoints for plotting path
            pts = info['pts']
            # Only draw path if plane has entered or entered recently
            if t > info['times'][0] and t < info['times'][3] + 60:
                # Approximate arc drawing
                cx, cy = info['center']
                # Tangent line
                ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], 'k:', alpha=0.1)
                # Final line
                ax.plot([pts[2][0], pts[3][0]], [pts[2][1], pts[3][1]], 'k:', alpha=0.1)
                # Circle (Full faint circle for reference)
                circle = plt.Circle((cx, cy), r, color='gray', fill=False, linestyle=':', alpha=0.1)
                ax.add_patch(circle)
        
        # Draw Aircraft
        plane_count = 0
        for idx, row in df_res.iterrows():
            eid = row['entry_id']
            pos = interpolate_position(traj_db[eid], t)
            
            if pos:
                plane_count += 1
                c = colors.get(row['corner'], 'black')
                
                # Marker
                ax.plot(pos[0], pos[1], 'o', color=c, markersize=8)
                
                # Label (ID)
                ax.text(pos[0]+0.5, pos[1]+0.5, f"{int(eid)}", fontsize=9, fontweight='bold')
                
                # Separation Check Visual
                # Find plane immediately ahead in landing sequence
                my_rank = row['landing_rank']
                if my_rank > 1:
                    prev_row = df_res[df_res['landing_rank'] == my_rank - 1].iloc[0]
                    prev_pos = interpolate_position(traj_db[prev_row['entry_id']], t)
                    
                    # Draw red line if too close (visual approximation of separation violation)
                    # Note: Real separation is Time-based at FAF, but distance check helps visual
                    if prev_pos:
                        dist = np.sqrt((pos[0]-prev_pos[0])**2 + (pos[1]-prev_pos[1])**2)
                        if dist < 2.5: # Visual warning for < 2.5nm proximity
                            ax.plot([pos[0], prev_pos[0]], [pos[1], prev_pos[1]], 'r-', linewidth=2)
        
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(f"Simulation Time: {t:.1f}s | Active Aircraft: {plane_count}")
        ax.set_xlabel("NM East")
        ax.set_ylabel("NM North")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Save Frame
        fname = f"{output_folder}/frame_{i:04d}.png"
        plt.savefig(fname)
        filenames.append(fname)
        plt.close(fig)
        
        if i % 20 == 0: print(f"Rendered frame {i}/{len(sim_times)}")
        
    print("Building GIF...")
    with imageio.get_writer('katl_arrival_optimization.gif', mode='I', fps=fps) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    print(f"GIF Saved: katl_arrival_optimization.gif")

# ==========================================
# RUN VISUALIZATION
# ==========================================
# Ensure df_final_hard has 'x_entry', 'y_entry', 'corner', 'is_north', 'long_arc'
# We might need to merge these back from df_arrivals if they were lost during processing
cols_needed = ['x_entry', 'y_entry', 'corner', 'is_north', 'long_arc', 'v_L', 'v_theta', 'v_f']
# Simple merge to ensure geometry columns exist
df_viz = pd.merge(df_final_hard, df_arrivals[['aircraft_id', 'x_entry', 'y_entry', 'corner', 'is_north', 'long_arc']], 
                  left_on='entry_id', right_on='aircraft_id', suffixes=('', '_orig'))

generate_scenario_gif(df_viz, fps=10, speed_up=10) # 10x speedup