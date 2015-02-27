#pragma once
#include <QtCore>
#include <Scene/Scene.h>
#include <Primitives/Puppet.h>

/************************************************************************/
/*                           IO Streams                                 */
/************************************************************************/
/*
* Material
*/
// Order: Name -> AmbientColor -> DiffuseColor -> SpecularColor -> Shininess -> Shininess Strength
//     -> Roughness -> Fresnel Reflectance -> Refractive Index -> Two Sided -> BlendMode -> AlphaBlending
QDataStream& operator << (QDataStream& out, MaterialPtr object)
{
	out << object->m_name << object->m_ambientColor << object->m_diffuseColor << object->m_specularColor << object->m_emissiveColor
		<< object->m_shininess << object->m_shininessStrength << object->m_roughness << object->m_fresnelReflectance
		<< object->m_refractiveIndex << object->m_twoSided << object->m_blendMode << object->m_alphaBlending;

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
	float refractiveIndex;

	int  twoSided;
	int  blendMode;
	bool alphaBlending;

	in >> name >> ambientColor >> diffuseColor >> specularColor >> emissiveColor
	   >> shininess >> shininessStrength >> roughness >> fresnelReflectance
	   >> refractiveIndex >> twoSided >> blendMode >> alphaBlending;

	object->m_name = name;
	object->m_ambientColor = ambientColor;
	object->m_diffuseColor = diffuseColor;
	object->m_specularColor = specularColor;
	object->m_emissiveColor = emissiveColor;
	object->m_shininess = shininess;
	object->m_shininessStrength = shininessStrength;
	object->m_roughness = roughness;
	object->m_fresnelReflectance = fresnelReflectance;
	object->m_refractiveIndex = refractiveIndex;
	object->m_twoSided = twoSided;
	object->m_blendMode = blendMode;
	object->m_alphaBlending = alphaBlending;

	return in;
}

/*
* Model (Only need out stream)
*/
// Order: File Name -> Shader File Name -> Number of Materials -> Each Material
QDataStream& operator << (QDataStream& out, ModelPtr object)
{
	out << object->fileName() << object->renderingEffect()->shaderFileName() << object->getMaterials().size();

	foreach(MaterialPtr mat, object->getMaterials())
	{
		out << mat;
	}

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
* Light
*/
// Order: Type -> Color -> Intensity -> Position -> Direction
//     -> Constant Attenuation -> Linear Attenuation -> Quadratic Attenuation
//     -> Spot Fall Off -> Spot Inner Angle -> Spot Outer Angle

QDataStream& operator << (QDataStream& out, LightPtr object)
{
	out << object->type() << object->color() << object->intensity() << object->position()
		<< object->direction() << object->constantAttenuation() << object->linearAttenuation()
		<< object->quadraticAttenuation() << object->spotFallOff() << object->spotInnerAngle()
		<< object->spotOuterAngle();

	return out;
}

QDataStream& operator >> (QDataStream& in, LightPtr object)
{
	int lightType;
	Light::LightType type;

	QColor col;
	float intensity, attConst, attLinear, attQuad, falloff, innerAngle, outerAngle;
	vec3 pos, dir;

	in >> lightType >> col >> intensity >> pos >> dir >> attConst >> attLinear >> attQuad
		>> falloff >> innerAngle >> outerAngle;

	switch(lightType)
	{
	case 0:
		type = Light::PointLight;
		break;
	case 1:
		type = Light::DirectionalLight;
		break;
	case 2:
		type = Light::SpotLight;
		break;
	case 3:
		type = Light::AmbientLight;
		break;
	case 4:
		type = Light::AreaLight;
		break;
	default:
		break;
	}

	object->setType(type);
	object->setColor(col);
	object->setIntensity(intensity);
	object->gameObject()->setPosition(pos);
	object->gameObject()->setRotation(dir);

	if (type == Light::PointLight || type == Light::SpotLight)
	{
		object->setAttenuation(attConst, attLinear, attQuad);
	}

	if (type == Light::SpotLight)
	{
		object->setSpotFalloff(falloff);
		object->setSpotInnerAngle(innerAngle);
		object->setSpotOuterAngle(outerAngle);
	}

	return in;
}

/*
* Rigid Body
*/
// Order: Mass -> Gravity Factor -> Restitution -> Linear Velocity
//     -> AngularVelocity -> Motion Type -> Size(radius or half extents, depends on the motion type)
QDataStream& operator << (QDataStream& out, RigidBodyPtr object)
{
	RigidBody::MotionType type = object->getMotionType();
	out << object->getMass() << object->getGravityFactor() << object->getRestitution() 
		<< object->getLinearVelocity() << object->getAngularVelocity() << type;

	if (type == RigidBody::MOTION_BOX_INERTIA)
	{
		BoxColliderPtr box = object->getBroadPhaseCollider().dynamicCast<BoxCollider>();
		out << box->getHalfExtents();
	}
	else if (type == RigidBody::MOTION_SPHERE_INERTIA)
	{
		SphereColliderPtr sphere = object->getBroadPhaseCollider().dynamicCast<SphereCollider>();
		out << sphere->getRadius();
	}

	return out;
}

QDataStream& operator >> (QDataStream& in, RigidBodyPtr object)
{
	int motionType;

	float mass, gravityFactor, restitution, radius;
	vec3 linearVelocity, angularVelocity, halfExtents;

	in >> mass >> gravityFactor >> restitution >> linearVelocity >> angularVelocity >> motionType;

	SphereColliderPtr sphere; 
	BoxColliderPtr box;

	switch(motionType)
	{
	case 0:
		in >> radius;
		object->setMotionType_SLOT("Sphere");
		sphere = object->getBroadPhaseCollider().dynamicCast<SphereCollider>();
		sphere->setRadius(radius);

		break;
	case 1:
		in >> halfExtents;
		object->setMotionType_SLOT("Box");
		box = object->getBroadPhaseCollider().dynamicCast<BoxCollider>();
		box->setHalfExtents(halfExtents);
		break;
	case 2:
		object->setMotionType_SLOT("Fixed");
		break;
	}

	object->setMass(mass);
	object->setGravityFactor(gravityFactor);
	object->setRestitution(restitution);
	object->setLinearVelocity(linearVelocity);
	object->setAngularVelocity(angularVelocity);

	return in;
}


/*
* Components (Only need Out stream)
*/
QDataStream& operator << (QDataStream& out, ComponentPtr object)
{
	if (object->className() == "Model")
	{
		ModelPtr model = object.dynamicCast<IModel>();
		out << model;
	}
	else if (object->className() == "ParticleSystem")
	{
		ParticleSystemPtr ps = object.dynamicCast<ParticleSystem>();
		out << ps;
	}
	else if (object->className() == "Light")
	{
		LightPtr light = object.dynamicCast<Light>();
		out << light;
	}
	else if (object->className() == "RigidBody")
	{
		RigidBodyPtr rb = object.dynamicCast<RigidBody>();
		out << rb;
	}
	
	return out;
}

/*
* Puppet Only need Out stream
*/
// Order: Variable -> Speed -> Duration 
QDataStream& operator << (QDataStream& out, PuppetPtr object)
{
	out << object->getVariable() << object->getSpeed() << object->getDuration();

	return out;
}


/*
* Game Object
*/
// Order: Object Name -> Transformation (Position -> Rotation -> Scaling) -> Components Count 
//     -> Component Type -> Each Component -> Puppets Count -> Each Puppet
QDataStream& operator << (QDataStream& out, GameObjectPtr object)
{
	out << object->objectName() << object->position() << object->rotation() << object->scale();
	
	QVector<ComponentPtr> components = object->getComponents();
	out << components.size();

	foreach(ComponentPtr comp, components)
	{
		out << comp->className() << comp;
	}

	QList<PuppetPtr> puppets = object->getPuppets();
	out << puppets.size();

	foreach(PuppetPtr p, puppets)
	{
		out << p;
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
			int materialCount;
			in >> fileName >> shaderName >> materialCount;
 			LoaderThread loader(object->getScene(), fileName, object, object->getScene()->sceneRoot(), false);

			// get the model
			ComponentPtr comp = object->getComponent("Model");
			ModelPtr model = comp.dynamicCast<IModel>();

			// apply the saved shader
			model->renderingEffect()->applyShader(shaderName);

			// change the materials
			QVector<MaterialPtr> mats = model->getMaterials();
			if (mats.size() != materialCount)
				qWarning() << "Materials doesn't match, did you change the model processing option?";

			for(int i = 0; i < mats.size(); ++i)
			{
				in >> mats[i];
			}
		}
		else if (className == "ParticleSystem")
		{
			// create a particle system and attach it to the this game object
			ParticleSystemPtr ps(new ParticleSystem(object->getScene()));
			object->attachComponent(ps);
			ps->initParticleSystem();
			in >> ps;
		}
		else if (className == "Light")
		{
			// create a light source
			LightPtr light(new Light(object->getScene(), object.data()));
			object->getScene()->addLight(light);
			object->attachComponent(light);
			in >> light;
		}
		else if (className == "RigidBody")
		{
			// create a rigid body and attach to this object
			RigidBodyPtr rb = object->getScene()->createRigidBody(object.data());
			in >> rb;
		}

	}

	// process puppets
	int puppetCount;
	in >> puppetCount;

	Puppet::Variable var;
	int variableType;
	vec3 speed;
	float duration;

	for (int i = 0; i < puppetCount; ++i)
	{
		in >> variableType >> speed >> duration;

		switch(variableType)
		{
		case 0:
			var = Puppet::Position;
			break;
		case 1:
			var = Puppet::Rotation;
			break;
		case 2:
			var = Puppet::Scale;
			break;
		case 3:
			var = Puppet::Color;
			break;
		default:
			break;
		}

		Puppet* p = new Puppet(object.data(), var, speed, duration);
	}

	return in;
}

/*
* Object Manager
*/
// Order: Game Objects Count -> Parent Name -> Each Game Object
QDataStream& operator << (QDataStream& out, ObjectManager* object)
{
	QList<GameObjectPtr> list = object->getGameObjectTreeList();
	out << list.size();

	foreach(GameObjectPtr go, list)
	{
		out << go->parent()->objectName() << go;
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
		QString parentName;
		in >> parentName;

		// find the parent object
		GameObjectPtr parent = object->getGameObject(parentName);

		GameObjectPtr go = object->getScene()->createEmptyGameObject("Game Object", parent.data());
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
// Order: Model Manager -> Camera -> SkyBox
QDataStream& operator << (QDataStream& out, Scene* object)
{
	out << object->objectManager() << object->getCamera() << object->isSkyBoxEnabled();

	return out;
}

QDataStream& operator >> (QDataStream& in, Scene* object)
{
	bool skyboxEnabled;
	in >> object->objectManager() >> object->getCamera() >> skyboxEnabled;

	object->toggleSkybox(skyboxEnabled);

	return in;
}