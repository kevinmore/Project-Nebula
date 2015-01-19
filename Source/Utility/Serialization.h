#pragma once
#include <QtCore>
#include <Scene/Scene.h>

/************************************************************************/
/*                           IO Streams                                 */
/************************************************************************/
/*
* Model (Only need out stream)
*/
// Order: File Name
QDataStream& operator << (QDataStream& out, ModelPtr object)
{
	out << object->fileName();
	return out;
}

/*
* Particle System
*/
// Order: Mass -> Gravity Factor -> Size -> Rate -> Amount -> MinLife -> MaxLife
//     -> Force -> CollisionEnabled -> Restitution -> MinVel -> MaxVel -> ColorRandom -> Color -> Texture File Name
QDataStream& operator << (QDataStream& out, ParticleSystemPtr object)
{
	out << object->getParticleMass() << object->getGravityFactor() << object->getParticleSize()
		<< object->getEmitRate() << object->getEmitAmount() << object->getMinLife()
		<< object->getMaxLife() << object->getForce() << object->isCollisionEnabled() << object->getRestitution()
		<<  object->getMinVel() << object->getMaxVel() << object->isColorRandom() 
		<< object->getParticleColor() << object->getTextureFileName();

	return out;
}

QDataStream& operator >> (QDataStream& in, ParticleSystemPtr object)
{
	float mass, gravitFactor, size, rate, minLife, maxLife, restitution;
	int amount;
	vec3 force, minVel, maxVel;
	bool colorRandom, collisionEnabled;
	QColor col;
	QString texFileName;

	in >> mass >> gravitFactor >> size >> rate >> amount >> minLife
	   >> maxLife >> force >> collisionEnabled >> restitution 
	   >> minVel >> maxVel >> colorRandom >> col >> texFileName;

	object->setParticleMass(mass);
	object->setGravityFactor(gravitFactor);
	object->setParticleSize(size);
	object->setEmitRate(rate);
	object->setEmitAmount(amount);
	object->setMinLife(minLife);
	object->setMaxLife(maxLife);
	object->setForce(force);
	object->toggleCollision(collisionEnabled);
	object->setRestitution(restitution);
	object->setMinVel(minVel);
	object->setMaxVel(maxVel);
	object->toggleRandomColor(colorRandom);
	object->setParticleColor(col);
	object->loadTexture(texFileName);

	return in;
}

/*
* Components (Model, Particle System) Only need Out stream
*/

QDataStream& operator << (QDataStream& out, ComponentPtr object)
{
	if (object->className() == "StaticModel" || object->className() == "RiggedModel")
	{
		ModelPtr model = object.dynamicCast<AbstractModel>();
		out << model;
	}
	else if (object->className() == "ParticleSystem")
	{
		ParticleSystemPtr ps = object.dynamicCast<ParticleSystem>();
		out << ps;
	}

	return out;
}

/*
* Game Object
*/
// Order: Object Name -> Transformation (Position -> Rotation -> Scaling) -> Components Count 
//     -> Component Type -> Each Component
QDataStream& operator << (QDataStream& out, GameObjectPtr object)
{
	out << object->objectName() << object->position() << object->rotation() << object->scale();

	QVector<ComponentPtr> components = object->getComponents();
	out << components.size();

	foreach(ComponentPtr comp, components)
	{
		out << comp->className() << comp;
	}

	return out;
}

QDataStream& operator >> (QDataStream& in, GameObjectPtr object)
{
	QString name;
	in >> name;
	object->setObjectName(name);

	vec3 pos, rot, scale;
	in >> pos >> rot >> scale;

	object->setPosition(pos);
	object->setRotation(rot);
	object->setScale(scale);

	// process the components
	int numComponents;
	in >> numComponents;

	for (int i = 0; i < numComponents; ++i)
	{
		QString className;
		in >> className;
		if (className == "StaticModel" || className == "RiggedModel")
		{
			// load a model and attach it to the this game object
			QString fileName;
			in >> fileName;
 			LoaderThread loader(object->getScene(), fileName, object.data(), object->getScene()->sceneNode(), false);
		}
		else if (className == "ParticleSystem")
		{
			// create a particle system and attach it to the this game object
			ParticleSystemPtr ps(new ParticleSystem(object->getScene()));
			object->attachComponent(ps);
			ps->initParticleSystem();

			in >> ps;
		}
	}

	return in;
}

/*
* Object Manager
*/
// Order: Game Objects Count -> Each Game Object
QDataStream& operator << (QDataStream& out, ObjectManagerPtr object)
{
	out << object->m_gameObjectMap.count();

	foreach(GameObjectPtr go, object->m_gameObjectMap)
	{
		out << go;
	}

	return out;
}

QDataStream& operator >> (QDataStream& in, ObjectManagerPtr object)
{
	int numGameObjects;
	in >> numGameObjects;

	// create each game object
	for (int i = 0; i < numGameObjects; ++i)
	{
		GameObjectPtr go = object->getScene()->createEmptyGameObject();
		QString autoName = go->objectName();
		in >> go;

		// since the object is already defined in the scene file
		// thus, we need to rename this game object
		// delete the current one
		go = object->m_gameObjectMap.take(autoName);
		// add the new record
		object->m_gameObjectMap[go->objectName()] = go;
	}

	return in;
}

/*
* Camera
*/
// Order: Position -> Up Vector -> View Center
QDataStream& operator << (QDataStream& out, Camera* object)
{
	out << object->position() << object->upVector() << object->viewCenter();

	return out;
}

QDataStream& operator >> (QDataStream& in, Camera* object)
{
	vec3 pos, up, center;

	in >> pos >> up >> center;

	object->setPosition(pos);
	object->setUpVector(up);
	object->setViewCenter(center);

	return in;
}

/*
* Scene
*/
// Order: Model Manager -> Camera
QDataStream& operator << (QDataStream& out, Scene* object)
{
	out << object->objectManager() << object->getCamera();

	return out;
}

QDataStream& operator >> (QDataStream& in, Scene* object)
{
	in >> object->objectManager() >> object->getCamera();

	return in;
}