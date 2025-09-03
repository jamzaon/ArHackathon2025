"""
Amazon Robotics Hackathon - Bokeh Visualizer

This module implements a Bokeh-based visualizer for the Amazon Robotics Hackathon game.
"""

import os
import networkx as nx
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource, HoverTool, Arrow, NormalHead, 
    Label, LabelSet, Range1d, Div, Slider, Button, CustomJS
)
from bokeh.layouts import column, row
from typing import Any, Dict, List, Tuple

from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package
from ar_hackathon.visualizers.base_visualizer import BaseVisualizer


class BokehVisualizer(BaseVisualizer):
    """
    Bokeh-based game visualizer for the Amazon Robotics Hackathon.
    
    This class handles the visualization of the game state using Bokeh.
    """
    
    def __init__(self):
        """Initialize the Bokeh visualizer."""
        # Initialize visualization parameters
        self.positions = None
        self.package_colors = {}
        self.package_markers = {}
        self.color_index = 0
        self.marker_types = ['circle', 'square', 'triangle', 'diamond', 'cross', 'x']
        
    def calculate_layout(self, fulfillment_centers, connections):
        """
        Calculate the layout for the fulfillment centers with edge lengths 
        proportional to weights.
        
        Args:
            fulfillment_centers: List of fulfillment centers
            connections: List of connections between fulfillment centers
            
        Returns:
            Dictionary mapping FC IDs to (x, y) positions
        """
        G = nx.DiGraph()
        
        # Add nodes
        for fc in fulfillment_centers:
            G.add_node(fc.id, name=fc.name)
        
        # Add edges with weights
        for conn in connections:
            G.add_edge(conn.from_fc, conn.to_fc, weight=conn.weight)
        
        # Use a layout algorithm that respects edge weights as distances
        # The weight parameter tells the algorithm to use the 'weight' edge attribute
        positions = nx.kamada_kawai_layout(G, weight='weight')
        
        # Store positions
        self.positions = positions
        
        return positions
    
    def create_frame(self, game_state: GameState, frame_number: int) -> Any:
        """
        Create a visualization frame for the current game state.
        
        Args:
            game_state: Current game state
            frame_number: Frame number
            
        Returns:
            Bokeh layout object representing the frame
        """
        # If positions haven't been calculated yet, do it now
        if self.positions is None:
            self.calculate_layout(game_state.fulfillment_centers, game_state.connections)
        
        # Create the main plot
        plot = figure(
            title=f"Amazon Robotics Hackathon - Time Step: {game_state.current_time_step}",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800, height=600,
            x_range=Range1d(-1.1, 1.1),
            y_range=Range1d(-1.1, 1.1)
        )
        
        # Remove grid and axes
        plot.grid.visible = False
        plot.axis.visible = False
        plot.outline_line_color = None
        
        # Add connections
        self._add_connections(plot, game_state)
        
        # Add fulfillment centers
        self._add_fulfillment_centers(plot, game_state)
        
        # Add packages
        self._add_packages(plot, game_state)
        
        # Create statistics panel
        stats_div = self._create_stats_panel(game_state)
        
        # Combine plot and stats in a layout
        layout = row(plot, stats_div)
        
        return layout
    
    def save_frame(self, frame: Any, output_dir: str, frame_number: int, 
                  save_html: bool, save_images: bool) -> None:
        """
        Save a frame as HTML and/or image.
        
        Args:
            frame: Bokeh layout to save
            output_dir: Directory to save to
            frame_number: Frame number
            save_html: Whether to save as HTML
            save_images: Whether to save as image
        """
        if save_html:
            html_path = os.path.join(output_dir, f"frame_{frame_number:04d}.html")
            output_file(html_path, title=f"Frame {frame_number}")
            save(frame)
        
        if save_images:
            # For static images, we need to use selenium to export
            try:
                from bokeh.io import export_png
                img_path = os.path.join(output_dir, f"frame_{frame_number:04d}.png")
                export_png(frame, filename=img_path)
            except Exception as e:
                print(f"Warning: Could not save PNG image. {str(e)}")
                print("To save PNG images, install selenium and a webdriver.")
    
    def create_animation(self, frames: List[Any], output_dir: str) -> None:
        """
        Create an animation from all frames.
        
        Args:
            frames: List of Bokeh layouts
            output_dir: Directory to save to
        """
        if not frames:
            return
        
        # Create a new figure for the animation
        plot = figure(
            title="Amazon Robotics Hackathon Animation",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=800, height=600,
            x_range=Range1d(-1.1, 1.1),
            y_range=Range1d(-1.1, 1.1)
        )
        
        # Remove grid and axes
        plot.grid.visible = False
        plot.axis.visible = False
        plot.outline_line_color = None
        
        # Create a slider to navigate between frames
        slider = Slider(start=0, end=len(frames)-1, value=0, step=1, title="Frame")
        
        # Create play/pause button
        play_button = Button(label="▶ Play", button_type="success")
        
        # Create a div for statistics
        stats_div = Div(text="", width=320, height=300)
        
        # Extract HTML content from each frame's stats div
        stats_html = []
        for frame in frames:
            stats_div_frame = frame.children[1]
            stats_html.append(stats_div_frame.text)
        
        # Create a callback to update the plot and stats
        callback = CustomJS(
            args=dict(plot=plot, stats_div=stats_div, stats_html=stats_html, slider=slider, button=play_button),
            code="""
            // Update the plot title
            plot.title.text = "Amazon Robotics Hackathon - Time Step: " + slider.value;
            
            // Update the stats div
            stats_div.text = stats_html[slider.value];
            
            // Auto-advance logic
            if (window.playing) {
                if (slider.value < slider.end) {
                    slider.value = slider.value + 1;
                } else {
                    window.playing = false;
                    button.label = "▶ Play";
                }
            }
            """
        )
        
        # Create a callback for the play button
        play_callback = CustomJS(
            args=dict(slider=slider, button=play_button),
            code="""
            if (!window.playing) {
                window.playing = true;
                button.label = "⏸ Pause";
                
                // Start auto-advance
                window.playInterval = setInterval(function() {
                    if (window.playing) {
                        if (slider.value < slider.end) {
                            slider.value = slider.value + 1;
                        } else {
                            window.playing = false;
                            button.label = "▶ Play";
                            clearInterval(window.playInterval);
                        }
                    }
                }, 500); // Advance every 500ms
            } else {
                window.playing = false;
                button.label = "▶ Play";
                clearInterval(window.playInterval);
            }
            """
        )
        
        # Add the callbacks
        slider.js_on_change('value', callback)
        play_button.js_on_click(play_callback)
        
        # Create a layout with the plot, slider, and button
        layout = column(
            row(play_button, slider),
            row(plot, stats_div)
        )
        
        # Add initialization code
        init_code = """
            // Initialize playing state
            window.playing = false;
        """
        
        layout.js_on_event('document_ready', CustomJS(code=init_code))
        
        # Save the animation
        animation_path = os.path.join(output_dir, "animation.html")
        output_file(animation_path, title="Animation")
        save(layout)
        
        print(f"Animation created at {animation_path}")
        print("Note: The animation currently shows only a placeholder. To view the actual frames, open the individual frame HTML files.")
    
    def _add_connections(self, plot, game_state):
        """Add connections to the plot with lengths proportional to weights."""
        for conn in game_state.connections:
            if conn.from_fc in self.positions and conn.to_fc in self.positions:
                start_pos = self.positions[conn.from_fc]
                end_pos = self.positions[conn.to_fc]
                
                # Determine color based on bandwidth
                if conn.bandwidth is not None:
                    if conn.available_bandwidth <= 0:
                        color = 'red'  # No bandwidth available
                    else:
                        # Scale from green (full) to yellow (half) to red (empty)
                        ratio = conn.available_bandwidth / conn.bandwidth
                        if ratio > 0.5:
                            # Green to yellow
                            r = 255 * (1 - ratio) * 2
                            g = 255
                            b = 0
                        else:
                            # Yellow to red
                            r = 255
                            g = 255 * ratio * 2
                            b = 0
                        color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
                else:
                    color = 'gray'  # Unlimited bandwidth
                
                # Add the connection line
                source = ColumnDataSource(data=dict(
                    x=[start_pos[0], end_pos[0]],
                    y=[start_pos[1], end_pos[1]],
                    from_fc=[conn.from_fc, conn.from_fc],
                    to_fc=[conn.to_fc, conn.to_fc],
                    weight=[conn.weight, conn.weight],
                    bandwidth=[
                        f"{conn.available_bandwidth}/{conn.bandwidth}" 
                        if conn.bandwidth is not None else "Unlimited",
                        f"{conn.available_bandwidth}/{conn.bandwidth}" 
                        if conn.bandwidth is not None else "Unlimited"
                    ]
                ))
                
                # Add the line
                line = plot.line(
                    'x', 'y', source=source,
                    line_width=2, color=color,
                    line_alpha=0.8
                )
                
                # Add arrow at the end
                arrow = Arrow(
                    end=NormalHead(size=10, fill_color=color),
                    x_start=start_pos[0], y_start=start_pos[1],
                    x_end=end_pos[0], y_end=end_pos[1],
                    line_width=0  # Hide the line as we already have one
                )
                plot.add_layout(arrow)
                
                # Add weight label
                label = Label(
                    x=(start_pos[0] + end_pos[0]) / 2,
                    y=(start_pos[1] + end_pos[1]) / 2,
                    text=str(conn.weight),
                    text_font_size='12pt',
                    text_color='black',
                    background_fill_color='white',
                    background_fill_alpha=0.7
                )
                plot.add_layout(label)
                
                # Add hover tool for the connection
                hover = HoverTool(
                    renderers=[line],  # Use the line renderer directly
                    tooltips=[
                        ("Connection", "@from_fc → @to_fc"),
                        ("Weight", "@weight"),
                        ("Bandwidth", "@bandwidth")
                    ]
                )
                plot.add_tools(hover)
    
    def _add_fulfillment_centers(self, plot, game_state):
        """Add fulfillment centers to the plot."""
        # Prepare data
        fc_x = []
        fc_y = []
        fc_id = []
        fc_name = []
        fc_packages = []
        
        for fc in game_state.fulfillment_centers:
            if fc.id in self.positions:
                x, y = self.positions[fc.id]
                fc_x.append(x)
                fc_y.append(y)
                fc_id.append(fc.id)
                fc_name.append(fc.name)
                
                # Count packages at this FC
                packages_at_fc = sum(1 for pkg in game_state.active_packages 
                                   if pkg.current_fc == fc.id and not pkg.in_transit)
                fc_packages.append(packages_at_fc)
        
        # Create data source
        source = ColumnDataSource(data=dict(
            x=fc_x, y=fc_y, id=fc_id, name=fc_name, packages=fc_packages
        ))
        
        # Add FCs as circles
        fc_circles = plot.scatter(
            'x', 'y', source=source,
            size=20, marker='circle', fill_color='lightblue', alpha=0.8,
            line_color='black', line_width=2
        )
        
        # Add FC labels
        labels = LabelSet(
            x='x', y='y', text='id',
            source=source,
            text_font_size='10pt',
            text_align='center',
            text_baseline='middle'
        )
        plot.add_layout(labels)
        
        # Add hover tool
        hover = HoverTool(
            renderers=[plot.renderers[-1]],  # The circles we just added
            tooltips=[
                ("FC", "@id"),
                ("Name", "@name"),
                ("Packages", "@packages")
            ]
        )
        plot.add_tools(hover)
    
    def _add_packages(self, plot, game_state):
        """Add packages to the plot with consistent colors/markers."""
        # Packages at FCs
        pkg_at_fc_x = []
        pkg_at_fc_y = []
        pkg_at_fc_id = []
        pkg_at_fc_current = []
        pkg_at_fc_dest = []
        pkg_at_fc_entry = []
        pkg_at_fc_color = []
        pkg_at_fc_marker = []
        
        # Packages in transit
        pkg_transit_x = []
        pkg_transit_y = []
        pkg_transit_id = []
        pkg_transit_from = []
        pkg_transit_to = []
        pkg_transit_dest = []
        pkg_transit_remaining = []
        pkg_transit_total = []
        pkg_transit_color = []
        pkg_transit_marker = []
        
        # Process active packages
        for pkg in game_state.active_packages:
            # Assign consistent color and marker if not already assigned
            if pkg.id not in self.package_colors:
                self.package_colors[pkg.id] = self._get_next_color()
                self.package_markers[pkg.id] = self._get_next_marker()
            
            if pkg.in_transit:
                # Calculate position along the path
                if pkg.current_fc in self.positions and pkg.transit_destination in self.positions:
                    start_pos = self.positions[pkg.current_fc]
                    end_pos = self.positions[pkg.transit_destination]
                    
                    # Find the connection to get the total transit time
                    connection = game_state.get_connection(pkg.current_fc, pkg.transit_destination)
                    if connection:
                        total_time = connection.weight
                        progress = 1 - (pkg.transit_remaining_time / total_time)
                        
                        # Linear interpolation
                        x = start_pos[0] + progress * (end_pos[0] - start_pos[0])
                        y = start_pos[1] + progress * (end_pos[1] - start_pos[1])
                        
                        pkg_transit_x.append(x)
                        pkg_transit_y.append(y)
                        pkg_transit_id.append(pkg.id)
                        pkg_transit_from.append(pkg.current_fc)
                        pkg_transit_to.append(pkg.transit_destination)
                        pkg_transit_dest.append(pkg.destination_fc)
                        pkg_transit_remaining.append(pkg.transit_remaining_time)
                        pkg_transit_total.append(total_time)
                        pkg_transit_color.append(self.package_colors[pkg.id])
                        pkg_transit_marker.append(self.package_markers[pkg.id])
            else:
                # Package is at an FC
                if pkg.current_fc in self.positions:
                    # Add a small offset to avoid overlapping with the FC
                    base_pos = self.positions[pkg.current_fc]
                    
                    # Count how many packages are at this FC to arrange them in a circle
                    packages_at_fc = [p for p in game_state.active_packages 
                                     if p.current_fc == pkg.current_fc and not p.in_transit]
                    
                    if len(packages_at_fc) > 1:
                        # Arrange packages in a circle around the FC
                        idx = packages_at_fc.index(pkg)
                        angle = 2 * np.pi * idx / len(packages_at_fc)
                        radius = 0.1  # Distance from FC center
                        offset_x = radius * np.cos(angle)
                        offset_y = radius * np.sin(angle)
                    else:
                        # Single package, place it slightly to the right of the FC
                        offset_x = 0.1
                        offset_y = 0.0
                    
                    x = base_pos[0] + offset_x
                    y = base_pos[1] + offset_y
                    
                    pkg_at_fc_x.append(x)
                    pkg_at_fc_y.append(y)
                    pkg_at_fc_id.append(pkg.id)
                    pkg_at_fc_current.append(pkg.current_fc)
                    pkg_at_fc_dest.append(pkg.destination_fc)
                    pkg_at_fc_entry.append(pkg.entry_time)
                    pkg_at_fc_color.append(self.package_colors[pkg.id])
                    pkg_at_fc_marker.append(self.package_markers[pkg.id])
        
        # Add packages at FCs
        if pkg_at_fc_x:
            source = ColumnDataSource(data=dict(
                x=pkg_at_fc_x, y=pkg_at_fc_y, id=pkg_at_fc_id,
                current=pkg_at_fc_current, dest=pkg_at_fc_dest,
                entry=pkg_at_fc_entry, color=pkg_at_fc_color,
                marker=pkg_at_fc_marker
            ))
            
            # We need to add each marker type separately
            for marker in set(pkg_at_fc_marker):
                indices = [i for i, m in enumerate(pkg_at_fc_marker) if m == marker]
                if indices:
                    # Create a separate data source for this marker type
                    marker_data = {
                        'x': [pkg_at_fc_x[i] for i in indices],
                        'y': [pkg_at_fc_y[i] for i in indices],
                        'id': [pkg_at_fc_id[i] for i in indices],
                        'current': [pkg_at_fc_current[i] for i in indices],
                        'dest': [pkg_at_fc_dest[i] for i in indices],
                        'entry': [pkg_at_fc_entry[i] for i in indices],
                        'color': [pkg_at_fc_color[i] for i in indices]
                    }
                    marker_source = ColumnDataSource(data=marker_data)
                    
                    # Add the markers
                    scatter = plot.scatter(
                        'x', 'y', source=marker_source,
                        size=10, marker=marker,
                        fill_color='color', line_color='black',
                        line_width=1
                    )
                    
                    # Add hover tool for this marker type
                    hover = HoverTool(
                        renderers=[scatter],
                        tooltips=[
                            ("Package", "@id"),
                            ("Current FC", "@current"),
                            ("Destination", "@dest"),
                            ("Entry Time", "@entry")
                        ]
                    )
                    plot.add_tools(hover)
        
        # Add packages in transit
        if pkg_transit_x:
            source = ColumnDataSource(data=dict(
                x=pkg_transit_x, y=pkg_transit_y, id=pkg_transit_id,
                from_fc=pkg_transit_from, to_fc=pkg_transit_to,
                dest=pkg_transit_dest, remaining=pkg_transit_remaining,
                total=pkg_transit_total, color=pkg_transit_color,
                marker=pkg_transit_marker
            ))
            
            # We need to add each marker type separately
            for marker in set(pkg_transit_marker):
                indices = [i for i, m in enumerate(pkg_transit_marker) if m == marker]
                if indices:
                    # Create a separate data source for this marker type
                    marker_data = {
                        'x': [pkg_transit_x[i] for i in indices],
                        'y': [pkg_transit_y[i] for i in indices],
                        'id': [pkg_transit_id[i] for i in indices],
                        'from_fc': [pkg_transit_from[i] for i in indices],
                        'to_fc': [pkg_transit_to[i] for i in indices],
                        'dest': [pkg_transit_dest[i] for i in indices],
                        'remaining': [pkg_transit_remaining[i] for i in indices],
                        'total': [pkg_transit_total[i] for i in indices],
                        'color': [pkg_transit_color[i] for i in indices]
                    }
                    marker_source = ColumnDataSource(data=marker_data)
                    
                    # Add the markers
                    scatter = plot.scatter(
                        'x', 'y', source=marker_source,
                        size=10, marker=marker,
                        fill_color='color', line_color='black',
                        line_width=1
                    )
                    
                    # Add hover tool for this marker type
                    hover = HoverTool(
                        renderers=[scatter],
                        tooltips=[
                            ("Package", "@id"),
                            ("From", "@from_fc"),
                            ("To", "@to_fc"),
                            ("Final Destination", "@dest"),
                            ("Remaining Time", "@remaining/@total")
                        ]
                    )
                    plot.add_tools(hover)
        
        # Add delivered packages count to legend
        delivered_count = len(game_state.delivered_packages)
        if delivered_count > 0:
            plot.scatter(
                x=[], y=[],  # No data points
                size=10, marker='circle', fill_color='blue',
                legend_label=f"Delivered Packages: {delivered_count}"
            )
    
    def _create_stats_panel(self, game_state):
        """Create a statistics panel."""
        # Calculate statistics
        active_packages = len(game_state.active_packages)
        delivered_packages = len(game_state.delivered_packages)
        total_packages = active_packages + delivered_packages
        delivery_percentage = (delivered_packages / total_packages * 100) if total_packages > 0 else 0
        
        # Calculate average delivery time
        if delivered_packages > 0:
            total_delivery_time = sum(pkg.delivery_time - pkg.entry_time for pkg in game_state.delivered_packages)
            average_delivery_time = total_delivery_time / delivered_packages
        else:
            average_delivery_time = 0
        
        # Count packages in transit
        packages_in_transit = sum(1 for pkg in game_state.active_packages if pkg.in_transit)
        
        # Create HTML for the stats panel
        html = f"""
        <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; width: 300px;">
            <h3>Statistics</h3>
            <table style="width: 100%;">
                <tr><td>Current Time Step</td><td>{game_state.current_time_step}</td></tr>
                <tr><td>Active Packages</td><td>{active_packages}</td></tr>
                <tr><td>Packages in Transit</td><td>{packages_in_transit}</td></tr>
                <tr><td>Delivered Packages</td><td>{delivered_packages}</td></tr>
                <tr><td>Total Packages</td><td>{total_packages}</td></tr>
                <tr><td>Delivery Percentage</td><td>{delivery_percentage:.2f}%</td></tr>
                <tr><td>Average Delivery Time</td><td>{average_delivery_time:.2f} steps</td></tr>
            </table>
        </div>
        """
        
        return Div(text=html, width=320, height=300)
    
    def _get_next_color(self):
        """Get the next color from a predefined palette."""
        # Define a color palette
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        color = colors[self.color_index % len(colors)]
        self.color_index += 1
        return color
    
    def _get_next_marker(self):
        """Get the next marker type."""
        marker = self.marker_types[self.color_index % len(self.marker_types)]
        return marker
