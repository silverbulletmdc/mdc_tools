import libtmux
def grid_search(command_dict, tmux_name='search'):
    """开一个tmux执行多个指令

    Args:
        command_dict (dict): 指令与窗口名
        tmux_name (str, optional): tmux session名. Defaults to 'search'.
    """
    server = libtmux.Server()
    sess = server.new_session(tmux_name)

    for i, (name, command_) in enumerate(command_dict.items()):
        if i == 0:
            window = sess.windows[-1]
        else:
            window = sess.new_window()
        window.rename_window(name)
        pane = window.panes[0]
        pane.send_keys(command_)
