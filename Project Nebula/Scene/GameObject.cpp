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

void GameObject::setSpeed( const QString& paramString )
{
	QStringList params = paramString.split(", ", QString::SkipEmptyParts);
	setSpeed(params[0].toFloat(), params[1].toFloat(), params[2].toFloat());
}

const QVector3D& GameObject::speed() const
{
	return m_speed;
}

void GameObject::rotate( const QString& paramString )
{
	QStringList params = paramString.split(", ", QString::SkipEmptyParts);
	QString axis   = params[0];
	float amount = params[1].toFloat();
	qDebug() << amount;
	if (axis.toLower() == "x") rotateX(m_rotation.x() + amount);
	else if (axis.toLower() == "y") rotateY(m_rotation.y() + amount);
	else if (axis.toLower() == "z") rotateZ(m_rotation.z() + amount);
}
