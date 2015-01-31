#include "GameObject.h"
#include <Scene/AbstractModel.h>
#include <Scene/Scene.h>
#include "Puppet.h"

GameObject::GameObject(Scene* scene, GameObject* parent)
	: QObject(parent),
	  m_scene(scene),
	  m_position(Vector3D::ZERO),
	  m_rotation(Vector3D::ZERO),
	  m_scale(Vector3D::UNIT_SCALE),
	  m_movingBehaviour(CONSECUTIVE),
	  m_modelMatrixDirty(true),
	  m_time(0.0f),
	  m_prevPosition(m_position),
	  m_isMoving(false)
{
	connect(this, SIGNAL(synchronized()), this, SLOT(calculateSpeed()));
	m_lifeTimer.start();
}

GameObject::~GameObject()
{
	m_components.clear();
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

void GameObject::fixedTranslateX(double x)
{
	m_position.setX(x);
	m_modelMatrixDirty = true;
}

void GameObject::fixedTranslateY(double y)
{
	m_position.setY(y);
	m_modelMatrixDirty = true;
}

void GameObject::fixedTranslateZ(double z)
{
	m_position.setZ(z);
	m_modelMatrixDirty = true;
}

void GameObject::fixedRotateX(double x)
{
	m_rotation.setX(x);
	m_modelMatrixDirty = true;
}

void GameObject::fixedRotateY(double y)
{
	m_rotation.setY(y);
	m_modelMatrixDirty = true;
}

void GameObject::fixedRotateZ(double z)
{
	m_rotation.setZ(z);
	m_modelMatrixDirty = true;
}

void GameObject::fixedScaleX(double x)
{
	m_scale.setX(x);
	m_modelMatrixDirty = true;
}

void GameObject::fixedScaleY(double y)
{
	m_scale.setY(y);
	m_modelMatrixDirty = true;
}

void GameObject::fixedScaleZ(double z)
{
	m_scale.setZ(z);
	m_modelMatrixDirty = true;
}

void GameObject::reset()
{
	setPosition(Vector3D::ZERO);
	setRotation(Vector3D::ZERO);
	setScale(Vector3D::UNIT_SCALE);
	m_puppets.clear();
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


QVector3D GameObject::globalSpeed() const
{
	quart rotX = quart::fromAxisAndAngle(Math::Vector3D::UNIT_X, m_rotation.x());
	quart rotY = quart::fromAxisAndAngle(Math::Vector3D::UNIT_Y, m_rotation.y());
	quart rotZ = quart::fromAxisAndAngle(Math::Vector3D::UNIT_Z, m_rotation.z());
	quart rot = rotX * rotY * rotZ;

	return rot.rotatedVector(m_speed);
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

	quart delta = quart(w, vec3(x, y, z));
	rotate(delta);
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

void GameObject::attachComponent( ComponentPtr pComponent )
{

	m_components.push_back(pComponent);
	pComponent->linkGameObject(this);

	emit componentAttached(pComponent);
}

QVector<ComponentPtr> GameObject::getComponents()
{
	return m_components;
}

ComponentPtr GameObject::getComponent( const QString& name )
{
	foreach(ComponentPtr comp, m_components)
	{
		if (comp->className() == name)
		{
			return comp;
		}
	}
	return ComponentPtr();
}

void GameObject::setScene( Scene* scene )
{
	m_scene = scene;
}

Scene* GameObject::getScene() const
{
	return m_scene;
}

void GameObject::translateX( float x )
{
	m_position.setX(m_position.x() + x);
	m_modelMatrix.translate(x, 0.0f, 0.0f);
	m_modelMatrixDirty = false;
}

void GameObject::translateY( float y )
{
	m_position.setY(m_position.y() + y);
	m_modelMatrix.translate(0.0f, y, 0.0f);
	m_modelMatrixDirty = false;
}

void GameObject::translateZ( float z )
{
	m_position.setZ(m_position.z() + z);
	m_modelMatrix.translate(0.0f, 0.0f, z);
	m_modelMatrixDirty = false;
}

void GameObject::translate( const vec3& delta )
{
	m_position += delta;
	m_modelMatrix.translate(delta);
	m_modelMatrixDirty = false;
}

void GameObject::rotateX( float x )
{
	m_rotation.setX(m_rotation.x() + x);
	m_modelMatrix.rotate(x, Math::Vector3D::UNIT_X);
	m_modelMatrixDirty = false;
}

void GameObject::rotateY( float y )
{
	m_rotation.setY(m_rotation.y() + y);
	m_modelMatrix.rotate(y, Math::Vector3D::UNIT_Y);
	m_modelMatrixDirty = false;
}

void GameObject::rotateZ( float z )
{
	m_rotation.setZ(m_rotation.z() + z);
	m_modelMatrix.rotate(z, Math::Vector3D::UNIT_Z);
	m_modelMatrixDirty = false;
}

void GameObject::rotate( const vec3& delta )
{
	m_rotation += delta;
	m_modelMatrix.rotate(delta.x(), Vector3D::UNIT_X);
	m_modelMatrix.rotate(delta.y(), Vector3D::UNIT_Y);
	m_modelMatrix.rotate(delta.z(), Vector3D::UNIT_Z);
	m_modelMatrixDirty = false;
}

void GameObject::rotate( const quart& delta )
{
	m_modelMatrix.rotate(delta);
	m_modelMatrixDirty = false;
}

void GameObject::scaleX( float x )
{
	m_scale.setX(m_scale.x() + x);
	m_modelMatrix.scale(m_scale);
	m_modelMatrixDirty = false;
}

void GameObject::scaleY( float y )
{
	m_scale.setY(m_scale.y() + y);
	m_modelMatrix.scale(m_scale);
	m_modelMatrixDirty = false;
}

void GameObject::scaleZ( float z )
{
	m_scale.setZ(m_scale.z() + z);
	m_modelMatrix.scale(m_scale);
	m_modelMatrixDirty = false;
}

void GameObject::scale( const vec3& delta )
{
	m_scale += delta;
	m_modelMatrix.scale(m_scale);
	m_modelMatrixDirty = false;
}

void GameObject::clearPuppets()
{
	m_puppets.clear();
}

void GameObject::addPuppet( PuppetPtr p )
{
	m_puppets << p;
}

void GameObject::removePuppet( Puppet* p )
{
	foreach(PuppetPtr pP, m_puppets)
	{
		if (pP.data() == p)
			m_puppets.removeOne(pP);
	}
}

void GameObject::toggleFill( bool state )
{
	foreach(ComponentPtr comp, m_components)
	{
		ModelPtr model = comp.dynamicCast<AbstractModel>();
		if (model)
		{
			model->setPolygonMode(AbstractModel::Fill);
		}
	}
}

void GameObject::toggleWireframe( bool state )
{
	foreach(ComponentPtr comp, m_components)
	{
		ModelPtr model = comp.dynamicCast<AbstractModel>();
		if (model)
		{
			model->setPolygonMode(AbstractModel::Line);
		}
	}
}

void GameObject::togglePoints( bool state )
{
	foreach(ComponentPtr comp, m_components)
	{
		ModelPtr model = comp.dynamicCast<AbstractModel>();
		if (model)
		{
			model->setPolygonMode(AbstractModel::Point);
		}
	}
}
