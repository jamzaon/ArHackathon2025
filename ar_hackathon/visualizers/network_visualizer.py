"""
Amazon Robotics Hackathon - Network Visualizer

This module implements a network-focused visualizer for the Amazon Robotics Hackathon game.
It emphasizes the network structure with connection weights, bandwidth utilization, and package counts.
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


class NetworkVisualizer(BaseVisualizer):
    """
    Network-focused game visualizer for the Amazon Robotics Hackathon.
    
    This class handles the visualization of the game state with emphasis on:
    - FC nodes with package counts
    - Connections with length proportional to weight
    - Connection color based on available bandwidth
    - Connection width based on package traffic
    - Interactive panning and zooming
    """
    
    # Visualization constants
    # Node appearance
    NODE_SIZE = 30
    NODE_COLOR = 'lightblue'
    NODE_LINE_WIDTH = 2
    NODE_LINE_COLOR = 'black'
    NODE_FONT_SIZE = 12
    
    # Connection appearance
    MIN_LINE_WIDTH = 1
    MAX_LINE_WIDTH = 10
    LINE_WIDTH_FACTOR = 1
    
    # Arrow appearance
    ARROW_SIZE = 0.03
    ARROW_POSITION = 0.85
    
    # Label appearance
    WEIGHT_LABEL_OFFSET = 0.05
    WEIGHT_LABEL_SIZE = 14
    PACKAGE_COUNT_LABEL_SIZE = 14
    
    # Package appearance
    PACKAGE_SIZE = 10
    PACKAGE_COLOR = 'blue'
    PACKAGE_AT_FC_SYMBOL = 'square'
    PACKAGE_IN_TRANSIT_SYMBOL = 'diamond'
    PACKAGE_CIRCLE_RADIUS = 0.1
    
    # Figure layout
    FIGURE_HEIGHT = 600
    FIGURE_WIDTH = 1200
    
    def __init__(self):
        """Initialize the Network visualizer."""
        # Initialize visualization parameters
        self.positions = None
        self.layout_seed = 42  # Fixed seed for consistent layouts
        
    def calculate_layout(self, fulfillment_centers: List[FulfillmentCenter], 
                        connections: List[Connection]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate the layout for the fulfillment centers with edge lengths 
        proportional to weights.
        
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
        
        # Add edges with weights
        for conn in connections:
            G.add_edge(conn.from_fc, conn.to_fc, weight=conn.weight)
        
        # Use a layout algorithm that respects edge weights as distances
        # The weight parameter tells the algorithm to use the 'weight' edge attribute
        # We use kamada_kawai_layout because it's good at preserving edge lengths
        # proportional to weights
        self.positions = nx.kamada_kawai_layout(G, weight='weight')
        
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
        self._add_network_to_figure(fig, game_state, frame_number)
        
        # Add statistics table
        self._add_stats_to_figure(fig, game_state)
        
        # Update layout
        self._update_figure_layout(fig, game_state)
        
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
            frame.write_html(html_path, config={
                'responsive': True,
                'scrollZoom': True,  # Enable scroll to zoom
                'displayModeBar': True,  # Always show the mode bar
                'modeBarButtonsToAdd': ['pan2d', 'zoom2d', 'resetScale2d']  # Add these buttons
            })
        
        if save_images:
            img_path = os.path.join(output_dir, f"frame_{frame_number:04d}.png")
            frame.write_image(img_path, width=self.FIGURE_WIDTH, height=self.FIGURE_HEIGHT)
    
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
        self._add_animation_controls(fig, len(frames))
        
        # Save animation with responsive configuration
        animation_path = os.path.join(output_dir, "animation.html")
        fig.write_html(animation_path, config={
            'responsive': True,
            'scrollZoom': True,  # Enable scroll to zoom
            'displayModeBar': True,  # Always show the mode bar
            'modeBarButtonsToAdd': ['pan2d', 'zoom2d', 'resetScale2d']  # Add these buttons
        })
    
    def _add_animation_controls(self, fig: go.Figure, frame_count: int) -> None:
        """
        Add animation controls (slider and buttons) to the figure.
        
        Args:
            fig: Plotly figure to add controls to
            frame_count: Number of frames in the animation
        """
        # Add slider
        sliders = [{
            'steps': [
                {
                    'method': 'animate',
                    'label': f"{i}",
                    'args': [[f"frame{i}"], {
                        'mode': 'immediate',
                        'frame': {'duration': 500, 'redraw': True},
                        'transition': {'duration': 0}
                    }]
                }
                for i in range(frame_count)
            ],
            'active': 0,
            'currentvalue': {"prefix": "Time Step: "}
        }]
        
        # Add play/pause buttons
        updatemenus = [{
            'type': 'buttons',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
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
    
    def _update_figure_layout(self, fig: go.Figure, game_state: GameState) -> None:
        """
        Update the figure layout with appropriate settings.
        
        Args:
            fig: Plotly figure to update
            game_state: Current game state
        """
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
            # height=self.FIGURE_HEIGHT,
            # width=self.FIGURE_WIDTH,
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
    
    def _add_network_to_figure(self, fig: go.Figure, game_state: GameState, frame_number: int = 0) -> None:
        """
        Add network visualization to the figure.
        
        Args:
            fig: Plotly figure to add to
            game_state: Current game state
            frame_number: Frame number for unique trace naming
        """
        # Draw connections first (so they appear behind nodes)
        self._draw_connections(fig, game_state)
        
        # Draw fulfillment centers
        self._draw_fulfillment_centers(fig, game_state)
        
        # Draw packages
        self._draw_packages(fig, game_state, frame_number)
    
    def _draw_connections(self, fig: go.Figure, game_state: GameState) -> None:
        """
        Draw connections between fulfillment centers.
        
        Args:
            fig: Plotly figure to add to
            game_state: Current game state
        """
        for conn in game_state.connections:
            if conn.from_fc in self.positions and conn.to_fc in self.positions:
                # Get connection endpoints
                x0, y0 = self.positions[conn.from_fc]
                x1, y1 = self.positions[conn.to_fc]
                
                # Get packages traversing this connection
                packages_on_connection = self._get_packages_on_connection(game_state, conn.from_fc, conn.to_fc)
                packages_count = len(packages_on_connection)
                
                # Determine line width based on packages traversing the connection
                line_width = self._calculate_connection_width(packages_count)
                
                # Determine color based on available bandwidth
                color = self._calculate_connection_color(conn)
                
                # Add the connection line
                self._draw_connection_line(fig, x0, y0, x1, y1, conn, packages_on_connection, line_width, color)
                
                # Add the arrowhead
                self._draw_connection_arrow(fig, x0, y0, x1, y1, color)
                
                # Add the weight label
                self._draw_weight_label(fig, x0, y0, x1, y1, conn.weight)
    
    def _calculate_connection_width(self, packages_count: int) -> float:
        """
        Calculate the width of a connection based on the number of packages traversing it.
        
        Args:
            packages_count: Number of packages traversing the connection
            
        Returns:
            Width of the connection line
        """
        line_width = self.MIN_LINE_WIDTH + (packages_count * self.LINE_WIDTH_FACTOR)
        return min(line_width, self.MAX_LINE_WIDTH)  # Cap at maximum width for aesthetics
    
    def _calculate_connection_color(self, conn: Connection) -> str:
        """
        Calculate the color of a connection based on its available bandwidth.
        
        Args:
            conn: Connection object
            
        Returns:
            Color string or RGB value
        """
        if conn.bandwidth is not None:
            if conn.available_bandwidth <= 0:
                return 'red'  # No bandwidth available
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
                return f'rgb({int(r)}, {int(g)}, {int(b)})'
        else:
            return 'gray'  # Unlimited bandwidth
    
    def _draw_connection_line(self, fig: go.Figure, x0: float, y0: float, x1: float, y1: float, 
                             conn: Connection, packages: List[Package], width: float, color: str) -> None:
        """
        Draw a connection line between two fulfillment centers.
        
        Args:
            fig: Plotly figure to add to
            x0, y0: Coordinates of the source FC
            x1, y1: Coordinates of the destination FC
            conn: Connection object
            packages: List of packages traversing this connection
            width: Width of the connection line
            color: Color of the connection line
        """
        # Format package IDs for tooltip
        package_ids_text = ""
        if packages:
            package_ids = [pkg.id for pkg in packages]
            package_ids_text = "<br>Package IDs: " + ", ".join(package_ids)
        
        # Add the edge line
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='text',
                hovertext=f"Connection: {conn.from_fc} → {conn.to_fc}<br>"
                         f"Weight: {conn.weight}<br>"
                         f"Bandwidth: {conn.available_bandwidth}/{conn.bandwidth if conn.bandwidth is not None else 'Unlimited'}<br>"
                         f"Packages in transit: {len(packages)}{package_ids_text}",
                showlegend=False
            ),
            row=1, col=1
        )
    
    def _draw_connection_arrow(self, fig: go.Figure, x0: float, y0: float, x1: float, y1: float, color: str) -> None:
        """
        Draw an arrow on a connection to indicate direction.
        
        Args:
            fig: Plotly figure to add to
            x0, y0: Coordinates of the source FC
            x1, y1: Coordinates of the destination FC
            color: Color of the arrow
        """
        # Calculate the position for the arrowhead
        ax = x0 + (x1 - x0) * self.ARROW_POSITION
        ay = y0 + (y1 - y0) * self.ARROW_POSITION

        # Calculate the angle of the connection line
        angle = np.arctan2(y1 - y0, x1 - x0)

        # Create a fixed equilateral triangle
        # Base triangle points (pointing right)
        triangle_x = [0, -self.ARROW_SIZE, -self.ARROW_SIZE, 0]
        triangle_y = [0, self.ARROW_SIZE/2, -self.ARROW_SIZE/2, 0]

        # Rotate the triangle to match the connection direction
        rotated_x = []
        rotated_y = []
        for i in range(len(triangle_x)):
            rotated_x.append(ax + triangle_x[i] * np.cos(angle) - triangle_y[i] * np.sin(angle))
            rotated_y.append(ay + triangle_x[i] * np.sin(angle) + triangle_y[i] * np.cos(angle))
        
        # Add the arrowhead
        fig.add_trace(
            go.Scatter(
                x=rotated_x,
                y=rotated_y,
                mode='lines',
                fill='toself',
                fillcolor=color,
                line=dict(color='black', width=1),
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=1
        )
    
    def _draw_weight_label(self, fig: go.Figure, x0: float, y0: float, x1: float, y1: float, weight: int) -> None:
        """
        Draw a weight label for a connection.
        
        Args:
            fig: Plotly figure to add to
            x0, y0: Coordinates of the source FC
            x1, y1: Coordinates of the destination FC
            weight: Weight of the connection
        """
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
        
        # Calculate the perpendicular vector for the label offset
        px = -dy
        py = dx
        
        # Add an offset perpendicular to the line to move the label away from the line
        label_x = mid_x + px * self.WEIGHT_LABEL_OFFSET
        label_y = mid_y + py * self.WEIGHT_LABEL_OFFSET
        
        # Add the weight label with improved visibility
        fig.add_trace(
            go.Scatter(
                x=[label_x],
                y=[label_y],
                mode='text',
                text=[str(weight)],
                textposition='middle center',
                textfont=dict(
                    size=self.WEIGHT_LABEL_SIZE,
                    color='black'
                ),
                # Add a white background to the text
                texttemplate='<span style="background-color: rgba(255,255,255,0.7); padding: 2px; border: 1px solid #cccccc;">%{text}</span>',
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=1
        )
    
    def _draw_fulfillment_centers(self, fig: go.Figure, game_state: GameState) -> None:
        """
        Draw fulfillment centers as nodes.
        
        Args:
            fig: Plotly figure to add to
            game_state: Current game state
        """
        node_x = []
        node_y = []
        node_text = []
        node_hover = []
        node_package_counts = []
        
        for fc in game_state.fulfillment_centers:
            if fc.id in self.positions:
                x, y = self.positions[fc.id]
                node_x.append(x)
                node_y.append(y)
                
                # Get packages at this FC
                packages_at_fc = [pkg for pkg in game_state.active_packages 
                                 if pkg.current_fc == fc.id and not pkg.in_transit]
                packages_count = len(packages_at_fc)
                
                # Format package IDs for tooltip
                package_ids_text = ""
                if packages_count > 0:
                    package_ids = [pkg.id for pkg in packages_at_fc]
                    package_ids_text = "<br>Package IDs: " + ", ".join(package_ids)
                
                node_text.append(fc.id)
                node_package_counts.append(packages_count)
                node_hover.append(
                    f"FC: {fc.id}<br>"
                    f"Name: {fc.name}<br>"
                    f"Packages: {packages_count}{package_ids_text}"
                )
        
        # Add FC nodes
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                marker=dict(
                    size=self.NODE_SIZE,
                    color=self.NODE_COLOR,
                    line=dict(width=self.NODE_LINE_WIDTH, color=self.NODE_LINE_COLOR)
                ),
                hoverinfo='text',
                hovertext=node_hover,
                name='Fulfillment Centers'
            ),
            row=1, col=1
        )
        
        # Add FC IDs
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='text',
                text=node_text,
                textposition="middle center",
                textfont=dict(
                    size=self.NODE_FONT_SIZE,
                    color='black'
                ),
                hoverinfo='skip',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add package counts as labels
        self._draw_package_count_labels(fig, node_x, node_y, node_package_counts)
    
    def _draw_package_count_labels(self, fig: go.Figure, node_x: List[float], 
                                  node_y: List[float], counts: List[int]) -> None:
        """
        Draw package count labels above fulfillment centers.
        
        Args:
            fig: Plotly figure to add to
            node_x: X-coordinates of FC nodes
            node_y: Y-coordinates of FC nodes
            counts: Package counts for each FC
        """
        for i, (x, y, count) in enumerate(zip(node_x, node_y, counts)):
            if count > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[x],
                        y=[y + 0.05],  # Position slightly above the node
                        mode='text',
                        text=[f"{count}"],
                        textposition="top center",
                        textfont=dict(
                            size=self.PACKAGE_COUNT_LABEL_SIZE,
                            color='black',
                            family='Arial Black'
                        ),
                        # Add a blue background to the text
                        texttemplate='<span style="background-color: rgba(173,216,230,0.7); padding: 3px; border-radius: 10px; border: 1px solid #0000FF;">%{text}</span>',
                        hoverinfo='skip',
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    def _draw_packages(self, fig: go.Figure, game_state: GameState, frame_number: int = 0) -> None:
        """
        Draw packages on the figure.
        
        Args:
            fig: Plotly figure to add to
            game_state: Current game state
            frame_number: Frame number for unique trace naming
        """
        # Draw packages at FCs
        self._draw_packages_at_fc(fig, game_state)
        
        # Draw packages in transit
        self._draw_packages_in_transit(fig, game_state, frame_number)
    
    def _draw_packages_at_fc(self, fig: go.Figure, game_state: GameState) -> None:
        """
        Draw packages that are currently at fulfillment centers.
        
        Args:
            fig: Plotly figure to add to
            game_state: Current game state
        """
        # Group packages by FC
        packages_by_fc = {}
        for pkg in game_state.active_packages:
            if not pkg.in_transit and pkg.current_fc in self.positions:
                if pkg.current_fc not in packages_by_fc:
                    packages_by_fc[pkg.current_fc] = []
                packages_by_fc[pkg.current_fc].append(pkg)
        
        # Draw packages for each FC
        for fc_id, packages in packages_by_fc.items():
            base_pos = self.positions[fc_id]
            
            # Calculate positions for each package
            positions = self._calculate_package_positions(base_pos, len(packages))
            
            # Prepare data for plotting
            pkg_x = []
            pkg_y = []
            pkg_hover = []
            
            for i, pkg in enumerate(packages):
                x, y = positions[i]
                pkg_x.append(x)
                pkg_y.append(y)
                pkg_hover.append(
                    f"Package: {pkg.id}<br>"
                    f"Current FC: {pkg.current_fc}<br>"
                    f"Destination: {pkg.destination_fc}<br>"
                    f"Entry Time: {pkg.entry_time}"
                )
            
            # Add packages to the figure
            if pkg_x:
                fig.add_trace(
                    go.Scatter(
                        x=pkg_x,
                        y=pkg_y,
                        mode='markers',
                        marker=dict(
                            size=self.PACKAGE_SIZE,
                            color=self.PACKAGE_COLOR,
                            symbol=self.PACKAGE_AT_FC_SYMBOL
                        ),
                        hoverinfo='text',
                        hovertext=pkg_hover,
                        name='Packages at FCs'
                    ),
                    row=1, col=1
                )
    
    def _draw_packages_in_transit(self, fig: go.Figure, game_state: GameState, frame_number: int = 0) -> None:
        """
        Draw packages that are currently in transit between fulfillment centers.
        
        Args:
            fig: Plotly figure to add to
            game_state: Current game state
            frame_number: Frame number for unique trace naming
        """
        pkg_x = []
        pkg_y = []
        pkg_hover = []
        
        for pkg in game_state.active_packages:
            if pkg.in_transit and pkg.current_fc in self.positions and pkg.transit_destination in self.positions:
                # Calculate position along the path
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
                    
                    pkg_x.append(x)
                    pkg_y.append(y)
                    pkg_hover.append(
                        f"Package: {pkg.id}<br>"
                        f"From: {pkg.current_fc}<br>"
                        f"To: {pkg.transit_destination}<br>"
                        f"Final Destination: {pkg.destination_fc}<br>"
                        f"Remaining Time: {pkg.transit_remaining_time}/{total_time}"
                    )
        
        # Add packages in transit to the figure
        if pkg_x:
            fig.add_trace(
                go.Scatter(
                    x=pkg_x,
                    y=pkg_y,
                    mode='markers',
                    marker=dict(
                        size=self.PACKAGE_SIZE,
                        color=self.PACKAGE_COLOR,
                        symbol=self.PACKAGE_IN_TRANSIT_SYMBOL
                    ),
                    hoverinfo='text',
                    hovertext=pkg_hover,
                    name=f'Packages in Transit - Frame {frame_number}'  # Add frame number to make unique
                ),
                row=1, col=1
            )
    
    def _calculate_package_positions(self, base_pos: Tuple[float, float], count: int) -> List[Tuple[float, float]]:
        """
        Calculate positions for packages around a fulfillment center.
        
        Args:
            base_pos: Base position (x, y) of the fulfillment center
            count: Number of packages to position
            
        Returns:
            List of (x, y) positions for each package
        """
        positions = []
        base_x, base_y = base_pos
        
        if count == 1:
            # Single package, place it slightly to the right of the FC
            positions.append((base_x + 0.1, base_y))
        else:
            # Multiple packages, arrange in a circle
            for i in range(count):
                angle = 2 * np.pi * i / count
                x = base_x + self.PACKAGE_CIRCLE_RADIUS * np.cos(angle)
                y = base_y + self.PACKAGE_CIRCLE_RADIUS * np.sin(angle)
                positions.append((x, y))
        
        return positions
    
    def _get_packages_on_connection(self, game_state: GameState, from_fc: str, to_fc: str) -> list:
        """
        Get the packages currently traversing a connection.
        
        Args:
            game_state: Current game state
            from_fc: Source FC ID
            to_fc: Destination FC ID
            
        Returns:
            List of packages on the connection
        """
        return [pkg for pkg in game_state.active_packages 
                if pkg.in_transit and pkg.current_fc == from_fc and pkg.transit_destination == to_fc]
    
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
