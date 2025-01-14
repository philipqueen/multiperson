import plotly.graph_objects as go
import numpy as np

def plot_3d_scatter(data_3d_dict: dict):
    # Determine axis limits based on the data
    all_data = np.concatenate(list(data_3d_dict.values()), axis=1)
    
    mean_x = np.nanmean(all_data[:, :, 0])
    mean_y = np.nanmean(all_data[:, :, 1])
    mean_z = np.nanmean(all_data[:, :, 2])

    ax_range = 2000


    # Create a Plotly figure
    fig = go.Figure()

    # Generate a frame for each time step
    frames = []
    for frame in range(all_data.shape[0]):
        frame_data = []
        for label, data in data_3d_dict.items():
            frame_data.append(go.Scatter3d(
                x=data[frame, :, 0],
                y=data[frame, :, 1],
                z=data[frame, :, 2],
                mode='markers',
                name=label,
                marker=dict(size=4, opacity=0.8)
            ))
        frames.append(go.Frame(data=frame_data, name=str(frame)))

    # Add the first frame's data
    for label, data in data_3d_dict.items():
        fig.add_trace(go.Scatter3d(
            x=data[0, :, 0],
            y=data[0, :, 1],
            z=data[0, :, 2],
            mode='markers',
            name=label,
            marker=dict(size=4, opacity=0.8),
            opacity=.7
        ))

    # Update the layout with sliders and other settings
    fig.update_layout(
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[mean_x - ax_range, mean_x + ax_range], title='X'),
            yaxis=dict(range=[mean_y - ax_range, mean_y + ax_range], title='Y'),
            zaxis=dict(range=[mean_z - ax_range, mean_z + ax_range], title='Z')
        ),
        title="3D Scatter Plot",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])]
        )],
        sliders=[{
            "steps": [{"args": [[str(frame)], {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}],
                       "label": str(frame), "method": "animate"} for frame in range(all_data.shape[0])],
            "currentvalue": {"prefix": "Frame: "}
        }]
    )

    # Add the frames to the figure
    fig.frames = frames

    # Show the plot
    fig.show()

if __name__ == "__main__":
    person_0_data_path = "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_16_32/recording_15_22_35_gmt-4__multiperson_no_contact_person0/output_data/yolo_skeleton_3d.npy"
    person_1_data_path = "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_16_32/recording_15_22_35_gmt-4__multiperson_no_contact_person1/output_data/yolo_skeleton_3d.npy"

    data_3d_dict = {
        "person_0": np.load(person_0_data_path),
        "person_1": np.load(person_1_data_path)
    }

    raw_data_path = "/Users/philipqueen/freemocap_data/recording_sessions/session_2024-06-27_15_16_32/recording_15_22_35_gmt-4__multiperson_no_contact/output_data/raw_data/yolo_3dData_numFrames_numTrackedPoints_spatialXYZ.npy"
    combined_data = np.concatenate([np.load(person_0_data_path), np.load(person_1_data_path)], axis=1)
    print(combined_data.shape)

    data_3d_dict = {
        "unmatched": np.load(raw_data_path),
        "matched": combined_data,
    }

    plot_3d_scatter(data_3d_dict)
