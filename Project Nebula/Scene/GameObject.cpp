#include "GameObject.h"
#include <Utility/Math.h>
using namespace Math;

// dae files, should rotate the model
GameObject::GameObject()
	: m_position(Vector3D::ZERO),
	  m_rotation(Vector3D::ZERO),
	  m_scale(Vector3D::UNIT_SCALE),
	  m_modelMatrixDirty(true)
{}

void GameObject::setPosition(const QVector3D& positionVector)
{
	m_position = positionVector;

	m_modelMatrixDirty = true;
}

void GameObject::setPosition(float x, float y, float z)
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

void GameObject::setRotation(float x, float y, float z)
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

void GameObject::setScale(float x, float y, float z)
{
	m_scale.setX(x);
	m_scale.setY(y);
	m_scale.setZ(z);

	m_modelMatrixDirty = true;
}

void GameObject::setScale(float scaleFactor)
{
	m_scale.setX(scaleFactor);
	m_scale.setY(scaleFactor);
	m_scale.setZ(scaleFactor);

	m_modelMatrixDirty = true;
}

void GameObject::translateX(float x)
{
	m_position.setX(x);
	m_modelMatrixDirty = true;
}

void GameObject::translateY(float y)
{
	m_position.setY(y);
	m_modelMatrixDirty = true;
}

void GameObject::translateZ(float z)
{
	m_position.setZ(z);
	m_modelMatrixDirty = true;
}

void GameObject::rotateX(float x)
{
	m_rotation.setX(x);
	m_modelMatrixDirty = true;
}

void GameObject::rotateY(float y)
{
	m_rotation.setY(y);
	m_modelMatrixDirty = true;
}

void GameObject::rotateZ(float z)
{
	m_rotation.setZ(z);
	m_modelMatrixDirty = true;
}

void GameObject::scaleX(float x)
{
	m_scale.setX(x);
	m_modelMatrixDirty = true;
}

void GameObject::scaleY(float y)
{
	m_scale.setY(y);
	m_modelMatrixDirty = true;
}

void GameObject::scaleZ(float z)
{
	m_scale.setZ(z);
	m_modelMatrixDirty = true;
}

const QVector3D& GameObject::position() const
{
	return m_position;
}

const QVector3D& GameObject::rotation() const
{
	return m_rotation;
}

const QVector3D& GameObject::scale() const
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

void GameObject::setObjectXPosition(int x)
{
	m_position.setX(static_cast<float>(x)/100.0f);
	m_modelMatrixDirty = true;
}

void GameObject::setObjectYPosition(int y)
{
	m_position.setY(static_cast<float>(y)/100.0f);
	m_modelMatrixDirty = true;
}

void GameObject::setObjectZPosition(int z)
{
	m_position.setZ(static_cast<float>(z)/100.0f);
	m_modelMatrixDirty = true;
}

void GameObject::setObjectXRotation(int x)
{
	m_rotation.setX(static_cast<float>(x));
	m_modelMatrixDirty = true;
}

void GameObject::setObjectYRotation(int y)
{
	m_rotation.setY(static_cast<float>(y));
	m_modelMatrixDirty = true;
}

void GameObject::setObjectZRotation(int z)
{
	m_rotation.setZ(static_cast<float>(z));
	m_modelMatrixDirty = true;
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

void GameObject::setSpeed( float x, float y, float z )
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

const QVector3D& GameObject::localSpeed() const
{
	return m_speed;
}

const QVector3D& GameObject::globalSpeed() const
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
	m_modelMatrixDirty = false;
}

void GameObject::translateInWorld( const QString& paramString )
{
	QStringList params = paramString.split(", ", QString::SkipEmptyParts);
	vec3 delta(params[0].toFloat(), params[1].toFloat(), params[2].toFloat());
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
