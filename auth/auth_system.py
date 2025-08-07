class SimpleAuthenticator:
    """Autenticação simples para demo e testes."""
    def __init__(self):
        self._users = {
            'admin': {'password': 'admin123', 'role': 'admin', 'name': 'Administrador'},
            'usuario': {'password': 'user123', 'role': 'user', 'name': 'Usuário'}
        }
        self._logged_in = None

    def login(self, username, password):
        user = self._users.get(username)
        if user and user['password'] == password:
            self._logged_in = username
            return True
        return False

    def logout(self):
        self._logged_in = None

    def is_logged_in(self):
        return self._logged_in is not None

    def get_user_info(self):
        if self._logged_in:
            user = self._users[self._logged_in]
            return {'username': self._logged_in, 'role': user['role'], 'name': user['name']}
        return {'username': None, 'role': None, 'name': None}
