#include "GameObject.h"
#include <Utility/Math.h>
#include <Scene/AbstractModel.h>
using namespace Math;

GameObject::GameObject(QObject* parent)
	: QObject(parent),
	  m_position(Vector3D::ZERO),
	  m_rotation(Vector3D::ZERO),
	  m_scale(Vector3D::UNIT_SCALE),
	  m_movingBehaviour(CONSECUTIVE),
	  m_modelMatrixDirty(true),
	  m_time(0.0f),
	  m_prevPosition(m_position),
	  m_isMoving(false),
	  m_model(NULL)
{
	connect(this, SIGNAL(synchronized()), this, SLOT(calculateSpeed()));
	m_lifeTimer.start();
}

GameObject::~GameObject()
{
}

void GameObject::setPosition(const QVector3D& positionVector)
{
	m_position = positionVector;

	m_modelMatrixDirty = true;
}

void GameObject::setPosition(double x, double y, double z)
{
	m_position.setX(x);
	m_position.setY(y);
	m_position.setZ(z);

	m_modelMatrixDirty = true;
}

void GameObject::setRotation(const QVector3D& rotationVector)
{
	m_rotation = rotationVector;

	m_modelMatrixDirty = true;
}

void GameObject::setRotation(double x, double y, double z)
{
	m_rotation.setX(x);
	m_rotation.setY(y);
	m_rotation.setZ(z);

	m_modelMatrixDirty = true;
}

void GameObject::setScale(const QVector3D& scale)
{
	m_scale = scale;

	m_modelMatrixDirty = true;
}

void GameObject::setScale(double x, double y, double z)
{
	m_scale.setX(x);
	m_scale.setY(y);
	m_scale.setZ(z);

	m_modelMatrixDirty = true;
}

void GameObject::setScale(double scaleFactor)
{
	m_scale.setX(scaleFactor);
	m_scale.setY(scaleFactor);
	m_scale.setZ(scaleFactor);

	m_modelMatrixDirty = true;
}

void GameObject::translateX(double x)
{
	m_position.setX(x);
	m_modelMatrixDirty = true;
}

void GameObject::translateY(double y)
{
	m_position.setY(y);
	m_modelMatrixDirty = true;
}

void GameObject::translateZ(double z)
{
	m_position.setZ(z);
	m_modelMatrixDirty = true;
}

void GameObject::rotateX(double x)
{
	m_rotation.setX(x);
	m_modelMatrixDirty = true;
}

void GameObject::rotateY(double y)
{
	m_rotation.setY(y);
	m_modelMatrixDirty = true;
}

void GameObject::rotateZ(double z)
{
	m_rotation.setZ(z);
	m_modelMatrixDirty = true;
}

void GameObject::scaleX(double x)
{
	m_scale.setX(x);
	m_modelMatrixDirty = true;
}

void GameObject::scaleY(double y)
{
	m_scale.setY(y);
	m_modelMatrixDirty = true;
}

void GameObject::scaleZ(double z)
{
	m_scale.setZ(z);
	m_modelMatrixDirty = true;
}

QVector3D GameObject::position() const
{
	return m_position;
}

QVector3D GameObject::rotation() const
{
	return m_rotation;
}

QVector3D GameObject::scale() const
{
	return m_scale;
}

const QMatrix4x4& GameObject::modelMatrix()
{
	if(m_modelMatrixDirty)
	{
		m_modelMatrix.setToIdentity();

		m_modelMatrix.translate(m_position);
		m_modelMatrix.rotate(m_rotation.x(), Vector3D::UNIT_X);
		m_modelMatrix.rotate(m_rotation.y(), Vector3D::UNIT_Y);
		m_modelMatrix.rotate(m_rotation.z(), Vector3D::UNIT_Z);
		m_modelMatrix.scale(m_scale);

		m_modelMatrixDirty = false;
	}

	return m_modelMatrix;
}

void GameObject::reset()
{
	setPosition(Vector3D::ZERO);
	setRotation(Vector3D::ZERO);
	setScale(Vector3D::UNIT_SCALE);
}

void GameObject::setSpeed( const QVector3D& speed )
{
	m_speed = speed;
}

void GameObject::setSpeed( double x, double y, double z )
{
	m_speed.setX(x);
	m_speed.setY(y);
	m_speed.setZ(z);
}

void GameObject::setLocalSpeed( const QString& paramString )
{
	QStringList params = paramString.split(", ", QString::SkipEmptyParts);
	setSpeed(params[0].toFloat(), params[1].toFloat(), params[2].toFloat());
}

QVector3D GameObject::localSpeed() const
{
	return m_speed;
}

QVector3D GameObject::globalSpeed() const
{
	QQuaternion rotX = QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_X, m_rotation.x());
	QQuaternion rotY = QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_Y, m_rotation.y());
	QQuaternion rotZ = QQuaternion::fromAxisAndAngle(Math::Vector3D::UNIT_Z, m_rotation.z());
	QQuaternion rot = rotX * rotY * rotZ;

	return rot.rotatedVector(m_speed);
}

void GameObject::rotateInWorld( const QQuaternion& delta )
{
	m_modelMatrix.rotate(delta);
	m_modelMatrixDirty = false;
}

void GameObject::translateInWorld( const QVector3D& delta )
{
	m_modelMatrix.translate(delta);
	m_prevPosition = m_position;
	m_position += delta;
	m_modelMatrixDirty = false;
}

void GameObject::translateInWorld( const QString& paramString )
{
	QStringList params = paramString.split(", ", QString::SkipEmptyParts);
	vec3 delta(params[0].toFloat(), params[1].toFloat(), params[2].toFloat());
	m_prevPosition = m_position;
	m_position += delta;
	m_modelMatrix.translate(delta);

	m_modelMatrixDirty = false;
	emit synchronized();
}

void GameObject::rotateInWorld( const QString& paramString )
{
	QStringList params = paramString.split(", ", QString::SkipEmptyParts);
	float w, x, y, z;

	w = params[0].toFloat();
	x = params[1].toFloat();
	y = params[2].toFloat();
	z = params[3].toFloat();

	QQuaternion delta = QQuaternion(w, vec3(x, y, z));
	rotateInWorld(delta);
	emit synchronized();
}

void GameObject::rotateInWorldAxisAndAngle( const QString& paramString )
{
	QStringList params = paramString.split(", ", QString::SkipEmptyParts);
	QString axis   = params[0];
	float amount = params[1].toFloat();
	if (axis.toLower() == "x") 
	{
		m_modelMatrix.rotate(amount, Math::Vector3D::UNIT_X);
		m_rotation.setX(m_rotation.x() + amount);
		if(m_rotation.x() > 180.0f) m_rotation.setX(m_rotation.x() - 360.0f);
		if(m_rotation.x() <= -180.0f) m_rotation.setX(m_rotation.x() + 360.0f);
	}
	else if (axis.toLower() == "y") 
	{
		m_modelMatrix.rotate(amount, Math::Vector3D::UNIT_Y);
		m_rotation.setY(m_rotation.y() + amount);
		if(m_rotation.y() > 180.0f) m_rotation.setY(m_rotation.y() - 360.0f);
		if(m_rotation.y() <= -180.0f) m_rotation.setY(m_rotation.y() + 360.0f);
	}
	else if (axis.toLower() == "z") 
	{
		m_modelMatrix.rotate(amount, Math::Vector3D::UNIT_Z);
		m_rotation.setZ(m_rotation.z() + amount);
		if(m_rotation.z() > 180.0f) m_rotation.setZ(m_rotation.z() - 360.0f);
		if(m_rotation.z() <= -180.0f) m_rotation.setZ(m_rotation.z() + 360.0f);
	}
	qDebug() << "rotation y" << m_rotation.y();
	m_modelMatrixDirty = false;
	emit synchronized();
}

void GameObject::setMovingBehaviour( MovingBehaviour type )
{
	m_movingBehaviour = type;
}

GameObject::MovingBehaviour GameObject::movingBehaviour() const
{
	return m_movingBehaviour;
}

void GameObject::calculateSpeed()
{
	float currentTime = (float)m_lifeTimer.elapsed()/1000;
	const float dt = currentTime - m_time;
	m_time = currentTime;
	if(dt < 0.5)
	{
		resetSpeed();
		return;
	}
	m_speed = (m_position - m_prevPosition) / dt;
}

bool GameObject::isMoving() const
{
	return m_isMoving;
}

void GameObject::setMoving( bool status )
{
	m_isMoving = status;
}

void GameObject::resetSpeed()
{
	setSpeed(0.0f, 0.0f, 0.0f);
}

QVector3D GameObject::predictedPosition()  const
{
	float currentTime = (float)m_lifeTimer.elapsed()/1000;
	const float dt = currentTime - m_time;
	return m_prevPosition + m_speed * dt;
}

void GameObject::attachModel( QSharedPointer<AbstractModel> pModel )
{
	m_model = pModel;
	pModel->linkGameObject(this);
}

QSharedPointer<AbstractModel> GameObject::getModel()
{
	return m_model;
}
