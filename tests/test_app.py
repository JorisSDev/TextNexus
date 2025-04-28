from app import app

def test_homepage_responds_correctly():
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b"TextNexus" in response.data

def test_settings_page():
    client = app.test_client()
    response = client.get('/settings')
    assert response.status_code == 200
    assert b"<html" in response.data

def test_users_page():
    client = app.test_client()
    response = client.get('/users')
    assert response.status_code == 200
    assert b"<html" in response.data

def test_login_page():
    client = app.test_client()
    response = client.get('/login')
    assert response.status_code == 200
    assert b"<html" in response.data

def test_register_page():
    client = app.test_client()
    response = client.get('/register')
    assert response.status_code == 200
    assert b"<html" in response.data

def test_info_page():
    client = app.test_client()
    response = client.get('/info')
    assert response.status_code == 200
    assert b"<html" in response.data

def test_model_configuration_page():
    client = app.test_client()
    response = client.get('/model_configuration')
    assert response.status_code == 200
    assert b"<html" in response.data

def test_dashboard_page():
    client = app.test_client()
    response = client.get('/dashboard')
    assert response.status_code == 200
    assert b"<html" in response.data