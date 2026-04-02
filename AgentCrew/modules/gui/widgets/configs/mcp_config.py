from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QLineEdit,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QStackedWidget,
)
from PySide6.QtCore import Qt, Signal

from AgentCrew.modules.config import ConfigManagement
from AgentCrew.modules.config.mcp_config import MCPConfig
from AgentCrew.modules.agents import AgentManager
from AgentCrew.modules.gui.widgets.json_editor import JsonEditor
from AgentCrew.modules.gui.themes import StyleProvider
from AgentCrew.modules.gui.widgets.loading_overlay import LoadingOverlay
from .save_worker import SaveWorker


class MCPsConfigTab(QWidget):
    """Tab for configuring MCP servers."""

    # Add signal for configuration changes
    config_changed = Signal()

    def __init__(self, config_manager: ConfigManagement):
        super().__init__()
        self.config_manager = config_manager
        self.agent_manager = AgentManager.get_instance()
        self.is_dirty = False  # Track unsaved changes
        self.is_code_view = False  # Track current view mode
        self.current_server_data = None  # Store current server data for view switching
        self.style_provider = StyleProvider()

        # Loading overlay for save operations
        self.save_worker = None

        # Load MCP configuration
        self.mcps_config = MCPConfig().read()

        self.init_ui()
        self.load_mcps()

        # Connect to theme changes
        self.style_provider.theme_changed.connect(self._on_theme_changed)

    def init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QHBoxLayout()

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - MCP server list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.mcps_list = QListWidget()
        self.mcps_list.currentItemChanged.connect(self.on_mcp_selected)

        # Buttons for MCP list management
        list_buttons_layout = QHBoxLayout()
        self.add_mcp_btn = QPushButton("Add")
        # Note: Need to access parent's style provider when the widget is parented
        # For now, use the main style constants
        style_provider = StyleProvider()
        self.add_mcp_btn.setStyleSheet(style_provider.get_button_style("primary"))
        self.add_mcp_btn.clicked.connect(self.add_new_mcp)
        self.remove_mcp_btn = QPushButton("Remove")
        self.remove_mcp_btn.setStyleSheet(style_provider.get_button_style("red"))
        self.remove_mcp_btn.clicked.connect(self.remove_mcp)
        self.remove_mcp_btn.setEnabled(False)  # Disable until selection

        list_buttons_layout.addWidget(self.add_mcp_btn)
        list_buttons_layout.addWidget(self.remove_mcp_btn)

        left_layout.addWidget(QLabel("MCP Servers:"))
        left_layout.addWidget(self.mcps_list)
        left_layout.addLayout(list_buttons_layout)

        # Right panel - MCP editor with view toggle
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        right_layout.addLayout(QHBoxLayout())  # Empty layout placeholder

        # Create stacked widget for switching between form and code views
        self.stacked_widget = QStackedWidget()

        # Form view (existing editor)
        form_scroll = QScrollArea()
        form_scroll.setWidgetResizable(True)

        self.editor_widget = QWidget()
        self.editor_widget.setStyleSheet(
            style_provider.get_editor_container_widget_style()
        )
        self.editor_layout = QVBoxLayout(self.editor_widget)

        # Form layout for MCP properties
        form_layout = QFormLayout()

        # Name field
        self.name_input = QLineEdit()
        self.name_input.textChanged.connect(self._mark_dirty)
        form_layout.addRow("Name:", self.name_input)

        # Streaming server checkbox
        self.streaming_server_checkbox = QCheckBox("Streaming Server")
        self.streaming_server_checkbox.stateChanged.connect(
            self._on_streaming_server_changed
        )
        form_layout.addRow("", self.streaming_server_checkbox)

        # URL field (for streaming servers)
        self.url_input = QLineEdit()
        self.url_input.textChanged.connect(self._mark_dirty)
        self.url_input.setPlaceholderText("http://localhost:8080/mcp")
        self.url_label = QLabel("URL:")
        form_layout.addRow(self.url_label, self.url_input)

        # Command field
        self.command_input = QLineEdit()
        self.command_input.textChanged.connect(self._mark_dirty)
        self.command_label = QLabel("Command:")
        form_layout.addRow(self.command_label, self.command_input)

        self.include_tools_input = QLineEdit()
        self.include_tools_input.textChanged.connect(self._mark_dirty)
        self.include_tools_input.setPlaceholderText("tool_a, tool_b")
        form_layout.addRow("Included Tools:", self.include_tools_input)

        # Arguments section
        args_group = QGroupBox("Arguments")
        self.args_group = args_group  # Store reference
        self.args_layout = QVBoxLayout()
        self.arg_inputs = []

        # Add button for arguments
        args_btn_layout = QHBoxLayout()
        self.add_arg_btn = QPushButton("Add Argument")
        self.add_arg_btn.setStyleSheet(style_provider.get_button_style("primary"))
        self.add_arg_btn.clicked.connect(lambda: self.add_argument_field(""))
        args_btn_layout.addWidget(self.add_arg_btn)
        args_btn_layout.addStretch()

        self.args_layout.addLayout(args_btn_layout)
        args_group.setLayout(self.args_layout)

        # Environment variables section
        env_group = QGroupBox("Environment Variables")
        self.env_group = env_group  # Store reference
        self.env_layout = QVBoxLayout()
        self.env_inputs = []

        # Add button for env vars
        env_btn_layout = QHBoxLayout()
        self.add_env_btn = QPushButton("Add Environment Variable")
        self.add_env_btn.setStyleSheet(style_provider.get_button_style("primary"))
        self.add_env_btn.clicked.connect(lambda: self.add_env_field("", ""))
        env_btn_layout.addWidget(self.add_env_btn)
        env_btn_layout.addStretch()

        self.env_layout.addLayout(env_btn_layout)
        env_group.setLayout(self.env_layout)

        # Headers section (for streaming servers)
        headers_group = QGroupBox("HTTP Headers")
        self.headers_group = headers_group  # Store reference
        self.headers_layout = QVBoxLayout()
        self.header_inputs = []

        # Add button for headers
        headers_btn_layout = QHBoxLayout()
        self.add_header_btn = QPushButton("Add Header")
        self.add_header_btn.setStyleSheet(style_provider.get_button_style("primary"))
        self.add_header_btn.clicked.connect(lambda: self.add_header_field("", ""))
        headers_btn_layout.addWidget(self.add_header_btn)
        headers_btn_layout.addStretch()

        self.headers_layout.addLayout(headers_btn_layout)
        headers_group.setLayout(self.headers_layout)

        # Enabled for agents section
        enabled_group = QGroupBox("Enabled For Agents")
        enabled_layout = QVBoxLayout()

        # Get available agents
        self.available_agents = list(self.agent_manager.agents.keys())

        self.agent_checkboxes = {}
        for agent in self.available_agents:
            checkbox = QCheckBox(agent)
            checkbox.stateChanged.connect(self._mark_dirty)
            self.agent_checkboxes[agent] = checkbox
            enabled_layout.addWidget(checkbox)

        enabled_group.setLayout(enabled_layout)

        # Add all components to editor layout (Save button moved to right_layout)
        self.editor_layout.addLayout(form_layout)
        self.editor_layout.addWidget(args_group)
        self.editor_layout.addWidget(env_group)
        self.editor_layout.addWidget(headers_group)
        self.editor_layout.addWidget(enabled_group)
        self.editor_layout.addStretch()

        form_scroll.setWidget(self.editor_widget)

        # Code view (JSON editor)
        self.json_editor = JsonEditor()
        self.json_editor.json_changed.connect(self._on_json_changed)
        self.json_editor.validation_error.connect(self._on_json_validation_error)

        # Add both views to stacked widget
        self.stacked_widget.addWidget(form_scroll)  # Index 0 - Form view
        self.stacked_widget.addWidget(self.json_editor)  # Index 1 - Code view

        right_layout.addWidget(self.stacked_widget)

        # Button layout with Show Code and Save buttons in same row
        button_layout = QHBoxLayout()

        # Show Code button (secondary color)
        self.show_code_btn = QPushButton("Show Code")
        self.show_code_btn.setStyleSheet(style_provider.get_button_style("secondary"))
        self.show_code_btn.clicked.connect(self._toggle_view_mode)
        self.show_code_btn.setEnabled(False)  # Disable until selection

        # Save button (primary color)
        self.save_btn = QPushButton("Save")
        self.save_btn.setStyleSheet(style_provider.get_button_style("primary"))
        self.save_btn.clicked.connect(self.save_mcp)
        self.save_btn.setEnabled(False)  # Disable until selection

        button_layout.addWidget(self.show_code_btn)
        button_layout.addWidget(self.save_btn)

        right_layout.addLayout(button_layout)

        # Add panels to splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 600])  # Initial sizes

        # Add splitter to main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Create loading overlay (will be parented to the main widget)
        self.loading_overlay = LoadingOverlay(self, "Saving MCP servers...")

        # Disable editor initially
        self.set_editor_enabled(False)

    def _mark_dirty(self, *args, **kwargs):
        """Mark the current configuration as dirty and update save button state."""
        # Check if the editor is supposed to be active for the current item
        if self.mcps_list.currentItem() and self.name_input.isEnabled():
            self.is_dirty = True
            self._update_save_button_state()

    def _update_save_button_state(self):
        """Enable or disable the save button based on current item and dirty state."""
        current_item_selected = self.mcps_list.currentItem() is not None
        can_save = current_item_selected and self.is_dirty
        self.save_btn.setEnabled(can_save)

    def _on_streaming_server_changed(self, state):
        """Handle streaming server checkbox state change."""
        is_streaming = state == Qt.CheckState.Checked.value

        # Show/hide fields and labels based on streaming server state
        self._set_sse_fields_visisble(is_streaming)
        self._set_stdio_fields_visible(not is_streaming)

        self._mark_dirty()

    def load_mcps(self):
        """Load MCP servers from configuration."""
        self.mcps_list.clear()

        for server_id, server_config in self.mcps_config.items():
            item = QListWidgetItem(server_config.get("name", server_id))
            item.setData(Qt.ItemDataRole.UserRole, (server_id, server_config))
            self.mcps_list.addItem(item)
        self.mcps_list.setCurrentRow(0)

    def _normalize_include_tools(self, include_tools) -> list[str]:
        if not isinstance(include_tools, list):
            return []

        normalized_tools = []
        seen_tools = set()
        for tool_name in include_tools:
            if not isinstance(tool_name, str):
                continue
            normalized_tool_name = tool_name.strip()
            if not normalized_tool_name or normalized_tool_name in seen_tools:
                continue
            normalized_tools.append(normalized_tool_name)
            seen_tools.add(normalized_tool_name)

        return normalized_tools

    def on_mcp_selected(self, current, previous):
        """Handle MCP server selection."""
        if current is None:
            self.set_editor_enabled(False)
            self.remove_mcp_btn.setEnabled(False)
            return

        # Enable editor and remove button
        self.set_editor_enabled(True)
        self.remove_mcp_btn.setEnabled(True)

        # Get MCP data
        server_id, server_config = current.data(Qt.ItemDataRole.UserRole)
        self.current_server_data = server_config

        # Reset to form view when switching items
        self.is_code_view = False
        self.stacked_widget.setCurrentIndex(0)
        self.show_code_btn.setText("Show Code")

        # Populate form
        self.name_input.setText(server_config.get("name", ""))
        self.streaming_server_checkbox.setChecked(
            server_config.get("streaming_server", False)
        )
        self.url_input.setText(server_config.get("url", ""))
        self.command_input.setText(server_config.get("command", ""))
        self.include_tools_input.setText(
            ", ".join(self._normalize_include_tools(server_config.get("includeTools")))
        )

        # Clear existing argument fields
        self.clear_argument_fields()

        # Add argument fields
        args = server_config.get("args", [])
        for arg in args:
            self.add_argument_field(arg, mark_dirty_on_add=False)

        # Clear existing env fields
        self.clear_env_fields()

        # Add env fields
        env = server_config.get("env", {})
        for key, value in env.items():
            self.add_env_field(key, value, mark_dirty_on_add=False)

        self.clear_header_fields()

        headers = server_config.get("headers", {})
        for key, value in headers.items():
            self.add_header_field(key, value, mark_dirty_on_add=False)

        # Set agent checkboxes
        enabled_agents = server_config.get("enabledForAgents", [])
        for agent, checkbox in self.agent_checkboxes.items():
            checkbox.setChecked(agent in enabled_agents)

        # Update field states based on streaming server
        self._on_streaming_server_changed(
            Qt.CheckState.Checked.value
            if server_config.get("streaming_server", False)
            else Qt.CheckState.Unchecked.value
        )

        self.is_dirty = False
        self._update_save_button_state()

    def _set_sse_fields_visisble(self, visible: bool):
        self.url_input.setVisible(visible)
        self.url_label.setVisible(visible)
        self.add_header_btn.setVisible(visible)
        for header_input in self.header_inputs:
            header_input["key_input"].setVisible(visible)
            header_input["value_input"].setVisible(visible)
            header_input["remove_btn"].setVisible(visible)
        if hasattr(self, "headers_group"):
            self.headers_group.setVisible(visible)

    def _set_stdio_fields_visible(self, visible: bool):
        self.command_input.setVisible(visible)
        self.command_label.setVisible(visible)
        self.add_arg_btn.setVisible(visible)
        self.add_env_btn.setVisible(visible)

        # Hide/show existing argument and env fields
        for arg_input in self.arg_inputs:
            arg_input["input"].setVisible(visible)
            arg_input["remove_btn"].setVisible(visible)

        for env_input in self.env_inputs:
            env_input["key_input"].setVisible(visible)
            env_input["value_input"].setVisible(visible)
            env_input["remove_btn"].setVisible(visible)

        if hasattr(self, "args_group"):
            self.args_group.setVisible(visible)
        if hasattr(self, "env_group"):
            self.env_group.setVisible(visible)

    def set_editor_enabled(self, enabled: bool):
        """Enable or disable the editor form."""
        self.name_input.setEnabled(enabled)
        self.streaming_server_checkbox.setEnabled(enabled)
        self.show_code_btn.setEnabled(enabled)
        self.include_tools_input.setEnabled(enabled)

        # For visibility-controlled fields, only disable them when editor is disabled
        # Their visibility is controlled by streaming_server state
        if enabled:
            is_streaming = self.streaming_server_checkbox.isChecked()
            self._set_stdio_fields_visible(not is_streaming)
            self._set_sse_fields_visisble(is_streaming)
        else:
            # When editor is disabled, hide all conditional fields and labels
            self.url_input.setVisible(False)
            self.url_label.setVisible(False)
            self.command_input.setVisible(False)
            self.command_label.setVisible(False)
            self.add_arg_btn.setVisible(False)
            self.add_env_btn.setVisible(False)
            if hasattr(self, "args_group"):
                self.args_group.setVisible(False)
            if hasattr(self, "env_group"):
                self.env_group.setVisible(False)
            if hasattr(self, "headers_group"):
                self.headers_group.setVisible(False)

        # Always enable/disable these regardless of visibility
        self.url_input.setEnabled(enabled)
        self.command_input.setEnabled(enabled)
        self.add_arg_btn.setEnabled(enabled)
        self.add_env_btn.setEnabled(enabled)
        self.add_header_btn.setEnabled(enabled)

        for checkbox in self.agent_checkboxes.values():
            checkbox.setEnabled(enabled)

        for arg_input in self.arg_inputs:
            arg_input["input"].setEnabled(enabled)
            arg_input["remove_btn"].setEnabled(enabled)
            if enabled:
                is_streaming = self.streaming_server_checkbox.isChecked()
                arg_input["input"].setVisible(not is_streaming)
                arg_input["remove_btn"].setVisible(not is_streaming)

        for env_input in self.env_inputs:
            env_input["key_input"].setEnabled(enabled)
            env_input["value_input"].setEnabled(enabled)
            env_input["remove_btn"].setEnabled(enabled)
            if enabled:
                is_streaming = self.streaming_server_checkbox.isChecked()
                env_input["key_input"].setVisible(not is_streaming)
                env_input["value_input"].setVisible(not is_streaming)
                env_input["remove_btn"].setVisible(not is_streaming)

        for header_input in self.header_inputs:
            header_input["key_input"].setEnabled(enabled)
            header_input["value_input"].setEnabled(enabled)
            header_input["remove_btn"].setEnabled(enabled)
            if enabled:
                is_streaming = self.streaming_server_checkbox.isChecked()
                header_input["key_input"].setVisible(is_streaming)
                header_input["value_input"].setVisible(is_streaming)
                header_input["remove_btn"].setVisible(is_streaming)

        self.json_editor.set_read_only(not enabled)

        if not enabled:
            self.is_dirty = False
            self.is_code_view = False
            self.stacked_widget.setCurrentIndex(0)  # Reset to form view
            self.show_code_btn.setText("Show Code")
        self._update_save_button_state()

    def add_argument_field(self, value="", mark_dirty_on_add=True):
        """Add a field for an argument."""
        arg_layout = QHBoxLayout()

        arg_input = QLineEdit()
        arg_input.setText(str(value))
        arg_input.textChanged.connect(self._mark_dirty)

        remove_btn = QPushButton("Remove")
        remove_btn.setMaximumWidth(80)

        style_provider = StyleProvider()
        remove_btn.setStyleSheet(style_provider.get_button_style("red"))

        arg_layout.addWidget(arg_input)
        arg_layout.addWidget(remove_btn)

        # Insert before the add button
        self.args_layout.insertLayout(len(self.arg_inputs), arg_layout)

        # Store references
        arg_data = {"layout": arg_layout, "input": arg_input, "remove_btn": remove_btn}
        self.arg_inputs.append(arg_data)

        # Connect remove button
        remove_btn.clicked.connect(lambda: self.remove_argument_field(arg_data))

        if mark_dirty_on_add:
            self._mark_dirty()
        return arg_data

    def remove_argument_field(self, arg_data):
        """Remove an argument field."""
        # Remove from layout
        self.args_layout.removeItem(arg_data["layout"])

        # Delete widgets
        arg_data["input"].deleteLater()
        arg_data["remove_btn"].deleteLater()

        # Remove from list
        self.arg_inputs.remove(arg_data)
        self._mark_dirty()

    def clear_argument_fields(self):
        """Clear all argument fields."""
        while self.arg_inputs:
            self.remove_argument_field(self.arg_inputs[0])

    def add_env_field(self, key="", value="", mark_dirty_on_add=True):
        """Add a field for an environment variable."""
        env_layout = QHBoxLayout()

        key_input = QLineEdit()
        key_input.setText(str(key))
        key_input.setPlaceholderText("Key")
        key_input.textChanged.connect(self._mark_dirty)

        value_input = QLineEdit()
        value_input.setText(str(value))
        value_input.setPlaceholderText("Value")
        value_input.textChanged.connect(self._mark_dirty)

        remove_btn = QPushButton("Remove")
        remove_btn.setMaximumWidth(80)

        style_provider = StyleProvider()
        remove_btn.setStyleSheet(style_provider.get_button_style("red"))

        env_layout.addWidget(key_input)
        env_layout.addWidget(value_input)
        env_layout.addWidget(remove_btn)

        # Insert before the add button
        self.env_layout.insertLayout(len(self.env_inputs), env_layout)

        # Store references
        env_data = {
            "layout": env_layout,
            "key_input": key_input,
            "value_input": value_input,
            "remove_btn": remove_btn,
        }
        self.env_inputs.append(env_data)

        # Connect remove button
        remove_btn.clicked.connect(lambda: self.remove_env_field(env_data))

        if mark_dirty_on_add:
            self._mark_dirty()
        return env_data

    def remove_env_field(self, env_data):
        """Remove an environment variable field."""
        # Remove from layout
        self.env_layout.removeItem(env_data["layout"])

        # Delete widgets
        env_data["key_input"].deleteLater()
        env_data["value_input"].deleteLater()
        env_data["remove_btn"].deleteLater()

        # Remove from list
        self.env_inputs.remove(env_data)
        self._mark_dirty()

    def clear_env_fields(self):
        """Clear all environment variable fields."""
        while self.env_inputs:
            self.remove_env_field(self.env_inputs[0])

    def add_header_field(self, key="", value="", mark_dirty_on_add=True):
        """Add a field for an HTTP header."""
        header_layout = QHBoxLayout()

        key_input = QLineEdit()
        key_input.setText(str(key))
        key_input.setPlaceholderText("Header Name (e.g., Authorization)")
        key_input.textChanged.connect(self._mark_dirty)

        value_input = QLineEdit()
        value_input.setText(str(value))
        value_input.setPlaceholderText("Header Value (e.g., Bearer token)")
        value_input.textChanged.connect(self._mark_dirty)

        remove_btn = QPushButton("Remove")
        remove_btn.setMaximumWidth(80)

        style_provider = StyleProvider()
        remove_btn.setStyleSheet(style_provider.get_button_style("red"))

        header_layout.addWidget(key_input)
        header_layout.addWidget(value_input)
        header_layout.addWidget(remove_btn)

        # Insert before the add button
        self.headers_layout.insertLayout(len(self.header_inputs), header_layout)

        # Store references
        header_data = {
            "layout": header_layout,
            "key_input": key_input,
            "value_input": value_input,
            "remove_btn": remove_btn,
        }
        self.header_inputs.append(header_data)

        # Connect remove button
        remove_btn.clicked.connect(lambda: self.remove_header_field(header_data))

        if mark_dirty_on_add:
            self._mark_dirty()
        return header_data

    def remove_header_field(self, header_data):
        """Remove an HTTP header field."""
        # Remove from layout
        self.headers_layout.removeItem(header_data["layout"])

        # Delete widgets
        header_data["key_input"].deleteLater()
        header_data["value_input"].deleteLater()
        header_data["remove_btn"].deleteLater()

        # Remove from list
        self.header_inputs.remove(header_data)
        self._mark_dirty()

    def clear_header_fields(self):
        """Clear all HTTP header fields."""
        while self.header_inputs:
            self.remove_header_field(self.header_inputs[0])

    def add_new_mcp(self):
        """Add a new MCP server to the configuration."""
        # Create a new server with default values
        server_id = f"new_server_{len(self.mcps_config) + 1}"
        new_server = {
            "name": "New Server",
            "command": "docker",
            "args": ["run", "-i", "--rm"],
            "env": {},
            "enabledForAgents": [],
            "streaming_server": False,
            "url": "",
            "headers": {},
            "includeTools": [],
        }

        # Add to list
        item = QListWidgetItem(new_server["name"])
        item.setData(Qt.ItemDataRole.UserRole, (server_id, new_server))
        self.mcps_list.addItem(item)
        self.mcps_list.setCurrentItem(item)

        # Mark as dirty since this is a new item that needs to be saved
        self.is_dirty = True
        self._update_save_button_state()

        # Focus on name field for immediate editing
        self.name_input.setFocus()
        self.name_input.selectAll()

    def remove_mcp(self):
        """Remove the selected MCP server."""
        current_item = self.mcps_list.currentItem()
        if not current_item:
            return

        server_id, server_config = current_item.data(Qt.ItemDataRole.UserRole)
        server_name = server_config.get("name", server_id)

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the MCP server '{server_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Remove from list
            row = self.mcps_list.row(current_item)
            self.mcps_list.takeItem(row)

            # Clear editor
            self.set_editor_enabled(False)
            self.name_input.clear()
            self.command_input.clear()
            self.include_tools_input.clear()
            self.clear_argument_fields()
            self.clear_env_fields()
            self.clear_header_fields()
            for checkbox in self.agent_checkboxes.values():
                checkbox.setChecked(False)
            self.save_all_mcps()

    def save_mcp(self):
        """Save the current MCP server configuration."""
        current_item = self.mcps_list.currentItem()
        if not current_item:
            return

        server_id, old_config = current_item.data(Qt.ItemDataRole.UserRole)

        if self.is_code_view:
            # Get data from JSON editor
            try:
                server_config = self.json_editor.get_json()
            except ValueError as e:
                QMessageBox.warning(
                    self,
                    "Invalid JSON",
                    f"Cannot save configuration: {str(e)}\nPlease fix the JSON syntax first.",
                )
                return
        else:
            # Get data from form
            server_config = self._get_form_data()

        # Extract values for validation
        name = server_config.get("name", "").strip()
        streaming_server = server_config.get("streaming_server", False)
        url = server_config.get("url", "").strip()
        command = server_config.get("command", "").strip()

        # Validate
        if not name:
            QMessageBox.warning(
                self, "Validation Error", "Server name cannot be empty."
            )
            return

        if streaming_server:
            if not url:
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "URL cannot be empty for streaming servers.",
                )
                return
        else:
            if not command:
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "Command cannot be empty for stdio servers.",
                )
                return

        current_item.setText(name)
        current_item.setData(Qt.ItemDataRole.UserRole, (server_id, server_config))

        if not self.is_code_view:
            # Refresh form to ensure consistency
            self._update_form_from_json(server_config, server_id)

        # Mark as clean since we just saved
        self.is_dirty = False
        self._update_save_button_state()

        # Save all servers to config
        self.save_all_mcps()

    def save_all_mcps(self):
        """Save all MCP servers to the configuration file with loading indicator."""
        self.loading_overlay.set_message("Saving MCP servers...")
        self.loading_overlay.show_loading()

        # Disable UI during save
        self.mcps_list.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.add_mcp_btn.setEnabled(False)
        self.remove_mcp_btn.setEnabled(False)
        self.show_code_btn.setEnabled(False)

        mcps_config = {}

        for i in range(self.mcps_list.count()):
            item = self.mcps_list.item(i)
            server_id, server_config = item.data(Qt.ItemDataRole.UserRole)
            mcps_config[server_id] = server_config

        self.mcps_config = mcps_config

        self.save_worker = SaveWorker(self._perform_mcp_save, self.mcps_config)
        self.save_worker.finished.connect(self._on_save_complete)
        self.save_worker.error.connect(self._on_save_error)
        self.save_worker.start()

    def _perform_mcp_save(self, mcps_config):
        """Perform the actual save operation (runs in worker thread)."""
        MCPConfig().write(mcps_config)

    def _on_save_complete(self):
        """Handle successful save completion."""
        self.loading_overlay.hide_loading()

        self.mcps_list.setEnabled(True)
        self.add_mcp_btn.setEnabled(True)

        if self.mcps_list.currentItem():
            self.remove_mcp_btn.setEnabled(True)
            self.show_code_btn.setEnabled(True)

        self.config_changed.emit()

        if self.save_worker:
            self.save_worker.deleteLater()
            self.save_worker = None

    def _on_save_error(self, error_message: str):
        """Handle save error."""
        # Hide loading overlay
        self.loading_overlay.hide_loading()

        self.mcps_list.setEnabled(True)
        self.add_mcp_btn.setEnabled(True)
        if self.mcps_list.currentItem():
            self.remove_mcp_btn.setEnabled(True)
            self.show_code_btn.setEnabled(True)

        QMessageBox.critical(
            self,
            "Save Error",
            f"Failed to save MCP servers configuration:\n{error_message}",
        )

        if self.save_worker:
            self.save_worker.deleteLater()
            self.save_worker = None

    def _toggle_view_mode(self):
        """Toggle between form view and code view."""
        if not self.mcps_list.currentItem():
            return

        current_item = self.mcps_list.currentItem()
        server_id, _ = current_item.data(Qt.ItemDataRole.UserRole)

        if self.is_code_view:
            try:
                json_data = self.json_editor.get_json()
                self._update_form_from_json(json_data, server_id)
                self.stacked_widget.setCurrentIndex(0)  # Form view
                self.show_code_btn.setText("Show Code")
                self.is_code_view = False
            except ValueError as e:
                QMessageBox.warning(
                    self,
                    "Invalid JSON",
                    f"Cannot switch to form view: {str(e)}\nPlease fix the JSON syntax first.",
                )
                return
        else:
            server_data = self._get_form_data()
            if server_data:  # Only proceed if form data is valid
                self.json_editor.set_json(server_data)
                self.stacked_widget.setCurrentIndex(1)  # Code view
                self.show_code_btn.setText("Show Form")
                self.is_code_view = True

    def _get_form_data(self) -> dict:
        """Get the current form data as a dictionary."""
        # Get values from form
        name = self.name_input.text().strip()
        streaming_server = self.streaming_server_checkbox.isChecked()
        url = self.url_input.text().strip()
        command = self.command_input.text().strip()
        include_tools = self._normalize_include_tools(
            self.include_tools_input.text().split(",")
        )

        # Get arguments
        args = []
        for arg_data in self.arg_inputs:
            arg_value = arg_data["input"].text().strip()
            if arg_value:
                args.append(arg_value)

        # Get environment variables
        env = {}
        for env_data in self.env_inputs:
            key = env_data["key_input"].text().strip()
            value = env_data["value_input"].text().strip()
            if key:
                env[key] = value

        # Get headers
        headers = {}
        for header_data in self.header_inputs:
            key = header_data["key_input"].text().strip()
            value = header_data["value_input"].text().strip()
            if key:
                headers[key] = value

        # Get enabled agents
        enabled_agents = [
            agent
            for agent, checkbox in self.agent_checkboxes.items()
            if checkbox.isChecked()
        ]

        return {
            "name": name,
            "command": command,
            "args": args,
            "env": env,
            "enabledForAgents": enabled_agents,
            "streaming_server": streaming_server,
            "url": url,
            "headers": headers,
            "includeTools": include_tools,
        }

    def _update_form_from_json(self, json_data: dict, server_id: str):
        """Update the form fields from JSON data."""
        current_item = self.mcps_list.currentItem()
        if current_item:
            current_item.setData(Qt.ItemDataRole.UserRole, (server_id, json_data))
            if "name" in json_data:
                current_item.setText(json_data["name"])

        self.name_input.setText(json_data.get("name", ""))
        self.streaming_server_checkbox.setChecked(
            json_data.get("streaming_server", False)
        )
        self.url_input.setText(json_data.get("url", ""))
        self.command_input.setText(json_data.get("command", ""))
        self.include_tools_input.setText(
            ", ".join(self._normalize_include_tools(json_data.get("includeTools")))
        )

        self.clear_argument_fields()
        for arg in json_data.get("args", []):
            self.add_argument_field(arg, mark_dirty_on_add=False)

        self.clear_env_fields()
        for key, value in json_data.get("env", {}).items():
            self.add_env_field(key, value, mark_dirty_on_add=False)

        self.clear_header_fields()
        for key, value in json_data.get("headers", {}).items():
            self.add_header_field(key, value, mark_dirty_on_add=False)

        enabled_agents = json_data.get("enabledForAgents", [])
        for agent, checkbox in self.agent_checkboxes.items():
            checkbox.setChecked(agent in enabled_agents)

        # Update field visibility based on streaming server
        self._on_streaming_server_changed(
            Qt.CheckState.Checked.value
            if json_data.get("streaming_server", False)
            else Qt.CheckState.Unchecked.value
        )

        # Mark as dirty since data changed
        self.is_dirty = True
        self._update_save_button_state()

    def _on_json_changed(self, json_data: dict):
        """Handle JSON editor content changes."""
        # Mark as dirty when JSON changes in code view
        if self.is_code_view:
            self.is_dirty = True
            self._update_save_button_state()

    def _on_json_validation_error(self, error_msg: str):
        """Handle JSON validation errors."""
        # Disable save button when JSON is invalid
        if self.is_code_view:
            self.save_btn.setEnabled(False)

    def _on_theme_changed(self, theme_name: str):
        """Handle theme changes by updating the JSON editor."""
        # Update JSON editor theme
        self.json_editor.update_theme()
