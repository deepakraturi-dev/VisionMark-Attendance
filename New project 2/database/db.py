from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()
login_manager = LoginManager()


def init_extensions(app):
    db.init_app(app)
    login_manager.login_view = "login"
    login_manager.login_message_category = "warning"
    login_manager.init_app(app)


def init_database(app):
    from database.models import User

    with app.app_context():
        db.create_all()
        admin = User.query.filter_by(email=app.config["ADMIN_EMAIL"]).first()
        if admin is None:
            admin = User(
                name=app.config["ADMIN_NAME"],
                email=app.config["ADMIN_EMAIL"],
                role="admin",
                active=True,
            )
            admin.set_password(app.config["ADMIN_PASSWORD"])
            db.session.add(admin)
            db.session.commit()
