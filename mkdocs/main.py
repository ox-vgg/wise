def define_env(env):
    repo_url = env.conf.get('repo_url', '').rstrip('/')
    repo_blob_infix = env.conf.get('extra', {}).get('repo_blob_infix', '/blob')
    @env.macro
    def repo_link(path, branch="main"):
        return f"[{path}]({repo_url}{repo_blob_infix}/{branch}/{path})"
