"""
Amazon Robotics Hackathon - Plotly Visualizer

This module implements a Plotly-based visualizer for the Amazon Robotics Hackathon game.
"""

import os
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple

from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package
from ar_hackathon.models.fulfillment_center import FulfillmentCenter
from ar_hackathon.models.connection import Connection
from ar_hackathon.visualizers.base_visualizer import BaseVisualizer


class PlotlyVisualizer(BaseVisualizer):
    """
    Plotly-based game visualizer for the Amazon Robotics Hackathon.
    
    This class handles the visualization of the game state using Plotly.
    """
    
    def __init__(self):
        """Initialize the Plotly visualizer."""
        # Initialize visualization parameters
        self.positions = None
        
        # Color mapping for packages - all the same color now
        self.package_colors = {
            'in_transit': 'blue',
            'at_fc': 'blue',
            'delivered': 'blue'
        }
    
    def calculate_layout(self, fulfillment_centers: List[FulfillmentCenter], 
                        connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate the layout for the fulfillment centers.
        
        Args:
            fulfillment_centers: List of fulfillment centers
            connections: List of connections between fulfillment centers
            
        Returns:
            Dictionary mapping FC IDs to (x, y) positions
        """
        # Create a networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        for fc in fulfillment_centers:
            G.add_node(fc.id, name=fc.name)
        
        # Add edges
        for conn in connections:
            G.add_edge(conn.from_fc, conn.to_fc, weight=conn.weight)
        
        # Calculate positions using spring layout
        self.positions = nx.spring_layout(G, seed=42)  # Fixed seed for consistency
        
        return self.positions
    
    def create_frame(self, game_state: GameState, frame_number: int) -> go.Figure:
        """
        Create a visualization frame for the current game state.
        
        Args:
            game_state: Current game state
            frame_number: Frame number
            
        Returns:
            Plotly figure representing the frame
        """
        # If positions haven't been calculated yet, do it now
        if self.positions is None:
            self.calculate_layout(game_state.fulfillment_centers, game_state.connections)
            
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            specs=[[{"type": "scatter"}, {"type": "table"}]],
            subplot_titles=["Network Visualization", "Statistics"]
        )
        
        # Add network visualization
        self._add_network_to_figure(fig, game_state)
        
        # Add statistics table
        self._add_stats_to_figure(fig, game_state)
        
        # Update layout
        fig.update_layout(
            title=f"Amazon Robotics Hackathon - Time Step: {game_state.current_time_step}",
            showlegend=False,
            legend=dict(
                title="Legend",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=600,
            width=1200,
            margin=dict(l=20, r=20, t=60, b=20),
            # Remove axes, grid, and background
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False
            ),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            paper_bgcolor='white',
            autosize=True  # Allow the figure to resize
        )
        
        return fig
    
    def save_frame(self, frame: go.Figure, output_dir: str, frame_number: int, 
                  save_html: bool, save_images: bool) -> None:
        """
        Save a frame as HTML and/or image.
        
        Args:
            frame: Plotly figure to save
            output_dir: Directory to save to
            frame_number: Frame number
            save_html: Whether to save as HTML
            save_images: Whether to save as image
        """
        if save_html:
            html_path = os.path.join(output_dir, f"frame_{frame_number:04d}.html")
            frame.write_html(html_path, config={'responsive': True})
        
        if save_images:
            img_path = os.path.join(output_dir, f"frame_{frame_number:04d}.png")
            frame.write_image(img_path)
    
    def create_animation(self, frames: List[go.Figure], output_dir: str) -> None:
        """
        Create an animation from all frames.
        
        Args:
            frames: List of Plotly figures
            output_dir: Directory to save to
        """
        if not frames:
            return
            
        # Create a figure with frames
        fig = frames[0]
        
        # Add frames
        fig.frames = [go.Frame(
            data=frame.data,
            layout=frame.layout,
            name=f"frame{i}"
        ) for i, frame in enumerate(frames)]
        
        # Add slider and buttons
        sliders = [{
            'steps': [
                {
                    'method': 'animate',
                    'label': f"{i}",
                    'args': [[f"frame{i}"], {
                        'mode': 'immediate',
                        'frame': {'duration': 500, 'redraw': True},
                        'transition': {'duration': 300}
                    }]
                }
                for i in range(len(frames))
            ],
            'active': 0,
            'currentvalue': {"prefix": "Time Step: "}
        }]
        
        updatemenus = [{
            'type': 'buttons',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 10},
            'showactive': False,
            'x': 0.1,
            'y': 0
        }]
        
        fig.update_layout(
            updatemenus=updatemenus,
            sliders=sliders
        )
        
        # Save animation with responsive configuration
        animation_path = os.path.join(output_dir, "animation.html")
        fig.write_html(animation_path, config={'responsive': True})
    
    def _add_network_to_figure(self, fig: go.Figure, game_state: GameState) -> None:
        """
        Add network visualization to the figure.
        
        Args:
            fig: Plotly figure to add to
            game_state: Current game state
        """
        # Add edges (connections)
        for conn in game_state.connections:
            if conn.from_fc in self.positions and conn.to_fc in self.positions:
                x0, y0 = self.positions[conn.from_fc]
                x1, y1 = self.positions[conn.to_fc]
                
                # Calculate the midpoint for the weight label
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                
                # Calculate the direction vector
                dx = x1 - x0
                dy = y1 - y0
                
                # Normalize the direction vector
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx /= length
                    dy /= length
                
                # Calculate the perpendicular vector for the arrowhead
                px = -dy
                py = dx
                
                # Add an offset perpendicular to the line to move the label away from the line
                offset_distance = 0.05  # Adjust this value as needed
                label_x = mid_x + px * offset_distance
                label_y = mid_y + py * offset_distance
                
                # Create a fixed equilateral triangle for the arrowhead
                arrow_size = 0.03
                arrow_position = 0.85  # Position at 85% along the line

                # Calculate the position for the arrowhead
                ax = x0 + (x1 - x0) * arrow_position
                ay = y0 + (y1 - y0) * arrow_position

                # Calculate the angle of the connection line
                angle = np.arctan2(y1 - y0, x1 - x0)

                # Create a fixed equilateral triangle
                # Base triangle points (pointing right)
                triangle_x = [0, -arrow_size, -arrow_size, 0]
                triangle_y = [0, arrow_size/2, -arrow_size/2, 0]

                # Rotate the triangle to match the connection direction
                rotated_x = []
                rotated_y = []
                for i in range(len(triangle_x)):
                    rotated_x.append(ax + triangle_x[i] * np.cos(angle) - triangle_y[i] * np.sin(angle))
                    rotated_y.append(ay + triangle_x[i] * np.sin(angle) + triangle_y[i] * np.cos(angle))
                
                # Determine line width based on weight
                line_width = max(1, min(10, 10 / conn.weight))
                
                # Determine color based on available bandwidth
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
                        color = f'rgb({int(r)}, {int(g)}, {int(b)})'
                else:
                    color = 'gray'  # Unlimited bandwidth
                
                # Add the edge line
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode='lines',
                        line=dict(width=line_width, color=color),
                        hoverinfo='text',
                        hovertext=f"Connection: {conn.from_fc} → {conn.to_fc}<br>"
                                 f"Weight: {conn.weight}<br>"
                                 f"Bandwidth: {conn.bandwidth if conn.bandwidth is not None else 'Unlimited'}<br>"
                                 f"Available: {conn.available_bandwidth if conn.bandwidth is not None else 'Unlimited'}<br>"
                                 f"Packages in transit: {self._get_packages_on_connection_text(game_state, conn.from_fc, conn.to_fc)}",
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Add the weight label with improved visibility
                fig.add_trace(
                    go.Scatter(
                        x=[label_x],
                        y=[label_y],
                        mode='text',
                        text=[str(conn.weight)],
                        textposition='middle center',
                        textfont=dict(
                            size=14,  # Increased from 10
                            color='black'
                        ),
                        # Add a white background to the text
                        texttemplate='<span style="background-color: rgba(255,255,255,0.7); padding: 2px; border: 1px solid #cccccc;">%{text}</span>',
                        hoverinfo='skip',
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Add nodes (fulfillment centers)
        node_x = []
        node_y = []
        node_text = []
        node_hover = []
        
        for fc in game_state.fulfillment_centers:
            if fc.id in self.positions:
                x, y = self.positions[fc.id]
                node_x.append(x)
                node_y.append(y)
                node_text.append(fc.id)
                
                # Get packages at this FC
                packages_at_fc = [pkg for pkg in game_state.active_packages 
                                 if pkg.current_fc == fc.id and not pkg.in_transit]
                packages_count = len(packages_at_fc)
                
                # Format package IDs for tooltip
                package_ids_text = ""
                if packages_count > 0:
                    package_ids = [pkg.id for pkg in packages_at_fc]
                    package_ids_text = "<br>Package IDs: " + ", ".join(package_ids)
                
                node_hover.append(
                    f"FC: {fc.id}<br>"
                    f"Name: {fc.name}<br>"
                    f"Packages: {packages_count}{package_ids_text}"
                )
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(
                    size=30,
                    color='lightblue',
                    line=dict(width=2, color='black')
                ),
                text=node_text,
                textposition="middle center",
                hoverinfo='text',
                hovertext=node_hover,
                name='Fulfillment Centers'
            ),
            row=1, col=1
        )
        
        # Add packages at FCs
        self._add_packages_to_figure(fig, game_state)
    
    def _get_packages_on_connection_text(self, game_state: GameState, from_fc: str, to_fc: str) -> str:
        """
        Get a formatted text of packages currently traversing a connection.
        
        Args:
            game_state: Current game state
            from_fc: Source FC ID
            to_fc: Destination FC ID
            
        Returns:
            Formatted text of packages on the connection
        """
        packages = [pkg for pkg in game_state.active_packages 
                   if pkg.in_transit and pkg.current_fc == from_fc and pkg.transit_destination == to_fc]
        
        if not packages:
            return "0"
        
        package_ids = [pkg.id for pkg in packages]
        return f"{len(packages)} ({', '.join(package_ids)})"
    
    def _add_packages_to_figure(self, fig: go.Figure, game_state: GameState) -> None:
        """
        Add packages to the figure.
        
        Args:
            fig: Plotly figure to add to
            game_state: Current game state
        """
        # Packages at FCs
        pkg_at_fc_x = []
        pkg_at_fc_y = []
        pkg_at_fc_hover = []
        
        # Packages in transit
        pkg_transit_x = []
        pkg_transit_y = []
        pkg_transit_hover = []
        
        # Process active packages
        for pkg in game_state.active_packages:
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
                        pkg_transit_hover.append(
                            f"Package: {pkg.id}<br>"
                            f"From: {pkg.current_fc}<br>"
                            f"To: {pkg.transit_destination}<br>"
                            f"Final Destination: {pkg.destination_fc}<br>"
                            f"Remaining Time: {pkg.transit_remaining_time}/{total_time}"
                        )
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
                    pkg_at_fc_hover.append(
                        f"Package: {pkg.id}<br>"
                        f"Current FC: {pkg.current_fc}<br>"
                        f"Destination: {pkg.destination_fc}<br>"
                        f"Entry Time: {pkg.entry_time}"
                    )
        
        # Add packages at FCs to the figure
        if pkg_at_fc_x:
            fig.add_trace(
                go.Scatter(
                    x=pkg_at_fc_x,
                    y=pkg_at_fc_y,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=self.package_colors['at_fc'],
                        symbol='square'
                    ),
                    hoverinfo='text',
                    hovertext=pkg_at_fc_hover,
                    name='Packages at FCs'
                ),
                row=1, col=1
            )
        
        # Add packages in transit to the figure
        if pkg_transit_x:
            fig.add_trace(
                go.Scatter(
                    x=pkg_transit_x,
                    y=pkg_transit_y,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=self.package_colors['in_transit'],
                        symbol='diamond'
                    ),
                    hoverinfo='text',
                    hovertext=pkg_transit_hover,
                    name='Packages in Transit'
                ),
                row=1, col=1
            )
        
        # Add delivered packages (just for the legend)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.package_colors['delivered'],
                    symbol='circle'
                ),
                name=f'Delivered Packages ({len(game_state.delivered_packages)})'
            ),
            row=1, col=1
        )
    
    def _add_stats_to_figure(self, fig: go.Figure, game_state: GameState) -> None:
        """
        Add statistics table to the figure.
        
        Args:
            fig: Plotly figure to add to
            game_state: Current game state
        """
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
        
        # Create table data
        table_data = [
            ["Current Time Step", game_state.current_time_step],
            ["Active Packages", active_packages],
            ["Packages in Transit", packages_in_transit],
            ["Delivered Packages", delivered_packages],
            ["Total Packages", total_packages],
            ["Delivery Percentage", f"{delivery_percentage:.2f}%"],
            ["Average Delivery Time", f"{average_delivery_time:.2f} steps"]
        ]
        
        # Add table to figure
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Statistic", "Value"],
                    fill_color='lightgrey',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color='white',
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=1, col=2
        )
