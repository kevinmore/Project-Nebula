#include "GameObject.h"
#include <Scene/IModel.h>
#include <Scene/Scene.h>
#include "Puppet.h"

GameObject::GameObject(Scene* scene, GameObject* parent)
	: QObject(parent),
	  m_scene(scene),
	  m_movingBehaviour(CONSECUTIVE),
	  m_transform(),
	  m_modelMatrixDirty(true),
	  m_time(0.0f),
	  m_prevPosition(Vector3::ZERO),
	  m_isMoving(false)
{
	connect(this, SIGNAL(synchronized()), this, SLOT(calculateSpeed()));

	m_lifeTimer.start();
}

GameObject::~GameObject()
{
	// remove the light from the light list of the scene
	foreach(ComponentPtr comp, m_components)
	{
		LightPtr l = comp.dynamicCast<Light>();
		if (l)
			m_scene->removeLight(l.data());
	}

	m_components.clear();
}

void GameObject::setPosition(const vec3& positionVector)
{
	m_transform.setPosition(positionVector);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setPosition(double x, double y, double z)
{
	m_transform.setPosition(x, y, z);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setRotation(const vec3& rotationVector)
{
	m_transform.setRotation(rotationVector);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setRotation(double x, double y, double z)
{
	m_transform.setRotation(x, y, z);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setScale(const vec3& scale)
{
	m_transform.setScale(scale);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setScale(double x, double y, double z)
{
	m_transform.setScale(x, y, z);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setScale(double scaleFactor)
{
	m_transform.setScale(scaleFactor);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setFixedPositionX(double x)
{
	m_transform.setPositionX(x);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setFixedPositionY(double y)
{
	m_transform.setPositionY(y);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setFixedPositionZ(double z)
{
	m_transform.setPositionZ(z);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setFixedRotationX(double x)
{
	m_transform.setEulerAngleX(x);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setFixedRotationY(double y)
{
	m_transform.setEulerAngleY(y);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setFixedRotationZ(double z)
{
	m_transform.setEulerAngleZ(z);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setFixedScaleX(double x)
{
	m_transform.setScaleX(x);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setFixedScaleY(double y)
{
	m_transform.setScaleY(y);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::setFixedScaleZ(double z)
{
	m_transform.setScaleZ(z);
	m_modelMatrixDirty = true;
	emit transformChanged(m_transform);
}

void GameObject::reset()
{
	m_transform.reset();
	m_puppets.clear();
	emit transformChanged(m_transform);
}

void GameObject::setSpeed( const vec3& speed )
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


vec3 GameObject::globalSpeed() const
{
	quat rotX = quat::fromAxisAndAngle(Math::Vector3::UNIT_X, m_transform.getEulerAngles().x());
	quat rotY = quat::fromAxisAndAngle(Math::Vector3::UNIT_Y, m_transform.getEulerAngles().y());
	quat rotZ = quat::fromAxisAndAngle(Math::Vector3::UNIT_Z, m_transform.getEulerAngles().z());
	quat rot = rotX * rotY * rotZ;

	return rot.rotatedVector(m_speed);
}

void GameObject::translateInWorld( const vec3& delta )
{
	m_modelMatrix.translate(delta);
	m_prevPosition = m_transform.getPosition();
	m_transform.setPosition(m_prevPosition + delta);
	m_modelMatrixDirty = false;
}

void GameObject::translateInWorld( const QString& paramString )
{
	QStringList params = paramString.split(", ", QString::SkipEmptyParts);
	vec3 delta(params[0].toFloat(), params[1].toFloat(), params[2].toFloat());
	m_prevPosition = m_transform.getPosition();
	m_transform.setPosition(m_prevPosition + delta);
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

	quat delta = quat(w, vec3(x, y, z));
	rotate(delta);
	emit synchronized();
}

// void GameObject::rotateInWorldAxisAndAngle( const QString& paramString )
// {
// 	QStringList params = paramString.split(", ", QString::SkipEmptyParts);
// 	QString axis   = params[0];
// 	float amount = params[1].toFloat();
// 	if (axis.toLower() == "x") 
// 	{
// 		m_modelMatrix.rotate(amount, Math::Vector3::UNIT_X);
// 		m_rotation.setX(m_rotation.x() + amount);
// 		if(m_rotation.x() > 180.0f) m_rotation.setX(m_rotation.x() - 360.0f);
// 		if(m_rotation.x() <= -180.0f) m_rotation.setX(m_rotation.x() + 360.0f);
// 	}
// 	else if (axis.toLower() == "y") 
// 	{
// 		m_modelMatrix.rotate(amount, Math::Vector3::UNIT_Y);
// 		m_rotation.setY(m_rotation.y() + amount);
// 		if(m_rotation.y() > 180.0f) m_rotation.setY(m_rotation.y() - 360.0f);
// 		if(m_rotation.y() <= -180.0f) m_rotation.setY(m_rotation.y() + 360.0f);
// 	}
// 	else if (axis.toLower() == "z") 
// 	{
// 		m_modelMatrix.rotate(amount, Math::Vector3::UNIT_Z);
// 		m_rotation.setZ(m_rotation.z() + amount);
// 		if(m_rotation.z() > 180.0f) m_rotation.setZ(m_rotation.z() - 360.0f);
// 		if(m_rotation.z() <= -180.0f) m_rotation.setZ(m_rotation.z() + 360.0f);
// 	}
// 	m_modelMatrixDirty = false;
// 	emit synchronized();
// }

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
	m_speed = (m_transform.getPosition() - m_prevPosition) / dt;
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

vec3 GameObject::predictedPosition()  const
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

void GameObject::detachComponent( ComponentPtr pComponent )
{
	int index = m_components.indexOf(pComponent);
	// make sure that the component is already attached
	if (index < 0) return;

	m_components.removeAt(index);
	pComponent->dislinkGameObject();

	emit componentDetached(pComponent);
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
	m_transform.setPositionX(m_transform.getPosition().x() + x);
	m_modelMatrix.translate(x, 0.0f, 0.0f);
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::translateY( float y )
{
	m_transform.setPositionX(m_transform.getPosition().y() + y);
	m_modelMatrix.translate(0.0f, y, 0.0f);
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::translateZ( float z )
{
	m_transform.setPositionZ(m_transform.getPosition().z() + z);
	m_modelMatrix.translate(0.0f, 0.0f, z);
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::translate( const vec3& delta )
{
	m_transform.setPosition(m_transform.getPosition() + delta);
	m_modelMatrix.translate(delta);
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::rotateX( float x )
{
	m_transform.setEulerAngleX(m_transform.getEulerAngles().x() + x);
	m_modelMatrix.rotate(x, Math::Vector3::UNIT_X);
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::rotateY( float y )
{
	m_transform.setEulerAngleY(m_transform.getEulerAngles().y() + y);
	m_modelMatrix.rotate(y, Math::Vector3::UNIT_Y);
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::rotateZ( float z )
{
	m_transform.setEulerAngleZ(m_transform.getEulerAngles().z() + z);
	m_modelMatrix.rotate(z, Math::Vector3::UNIT_Z);
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::rotate( const vec3& delta )
{
	m_transform.setRotation(m_transform.getEulerAngles() + delta);
	m_modelMatrix.rotate(delta.x(), Vector3::UNIT_X);
	m_modelMatrix.rotate(delta.y(), Vector3::UNIT_Y);
	m_modelMatrix.rotate(delta.z(), Vector3::UNIT_Z);
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::rotate( const quat& delta )
{
	m_modelMatrix.rotate(delta);
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::scaleX( float x )
{
	m_transform.setScaleX(m_transform.getScale().x() + x);
	m_modelMatrix.scale(m_transform.getScale());
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::scaleY( float y )
{
	m_transform.setScaleY(m_transform.getScale().y() + y);
	m_modelMatrix.scale(m_transform.getScale());
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::scaleZ( float z )
{
	m_transform.setScaleZ(m_transform.getScale().z() + z);
	m_modelMatrix.scale(m_transform.getScale());
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::scale( const vec3& delta )
{
	m_transform.setScale(m_transform.getScale() + delta);
	m_modelMatrix.scale(m_transform.getScale());
	m_modelMatrixDirty = false;
	emit transformChanged(m_transform);
}

void GameObject::clearPuppets()
{
	m_puppets.clear();
}

void GameObject::addPuppet( PuppetPtr p )
{
	m_puppets << p;
}

QList<PuppetPtr> GameObject::getPuppets()
{
	return m_puppets;
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
		ModelPtr model = comp.dynamicCast<IModel>();
		if (model)
		{
			model->setPolygonMode(IModel::Fill);
		}
	}
}

void GameObject::toggleWireframe( bool state )
{
	foreach(ComponentPtr comp, m_components)
	{
		ModelPtr model = comp.dynamicCast<IModel>();
		if (model)
		{
			model->setPolygonMode(IModel::Line);
		}
	}
}

void GameObject::togglePoints( bool state )
{
	foreach(ComponentPtr comp, m_components)
	{
		ModelPtr model = comp.dynamicCast<IModel>();
		if (model)
		{
			model->setPolygonMode(IModel::Point);
		}
	}
}

QStringList GameObject::getComponentsTypes()
{
	QStringList types;
	foreach(ComponentPtr comp, m_components)
	{
		QString t = comp->className();
		if(types.count(t) == 0)
			types << t;
	}

	return types;
}
