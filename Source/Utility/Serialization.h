#pragma once
#include <QtCore>
#include <Scene/Scene.h>

/************************************************************************/
/*                           IO Streams                                 */
/************************************************************************/
/*
* Model (Only need out stream)
*/
// Order: File Name -> Shader File Name
QDataStream& operator << (QDataStream& out, ModelPtr object)
{
	out << object->fileName() << object->renderingEffect()->shaderFileName();
	return out;
}

/*
* Material
*/
// Order: Name -> AmbientColor -> DiffuseColor -> SpecularColor -> Shininess -> Shininess Strength
//     -> Roughness -> Fresnel Reflectance -> Two Sided -> BlendMode -> AlphaBlending
QDataStream& operator << (QDataStream& out, MaterialPtr object)
{
	out << object->m_name << object->m_ambientColor << object->m_diffuseColor << object->m_specularColor << object->m_emissiveColor
		<< object->m_shininess << object->m_shininessStrength << object->m_roughness << object->m_fresnelReflectance
		<< object->m_twoSided << object->m_blendMode << object->m_alphaBlending;

	return out;
}

QDataStream& operator >> (QDataStream& in, MaterialPtr object)
{
	QString name;

	QColor ambientColor;
	QColor diffuseColor;
	QColor specularColor;
	QColor emissiveColor;

	float shininess;
	float shininessStrength;
	float roughness;
	float fresnelReflectance;

	int  twoSided;
	int  blendMode;
	bool alphaBlending;

	in >> name >> ambientColor >> diffuseColor >> specularColor >> emissiveColor
	   >> shininess >> shininessStrength >> roughness >> fresnelReflectance
	   >> twoSided >> blendMode >> alphaBlending;

	object->m_name = name;
	object->m_ambientColor = ambientColor;
	object->m_diffuseColor = diffuseColor;
	object->m_specularColor = specularColor;
	object->m_emissiveColor = emissiveColor;
	object->m_shininess = shininess;
	object->m_shininessStrength = shininessStrength;
	object->m_roughness = roughness;
	object->m_fresnelReflectance = fresnelReflectance;
	object->m_twoSided = twoSided;
	object->m_blendMode = blendMode;
	object->m_alphaBlending = alphaBlending;

	return in;
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
	if (object->className() == "Model")
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
		if (className == "Model")
		{
			// load a model and attach it to the this game object
			QString fileName, shaderName;
			in >> fileName >> shaderName;
 			LoaderThread loader(object->getScene(), fileName, object, object->getScene()->sceneRoot(), false);
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
QDataStream& operator << (QDataStream& out, ObjectManager* object)
{
	out << object->m_gameObjectMap.count();

	foreach(GameObjectPtr go, object->m_gameObjectMap)
	{
		out << go;
	}

	return out;
}

QDataStream& operator >> (QDataStream& in, ObjectManager* object)
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