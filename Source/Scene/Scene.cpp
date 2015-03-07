#include "Scene.h"
#include <UI/Canvas.h>
#include <Utility/LoaderThread.h>
#include <Utility/Serialization.h>
#include <Primitives/Puppet.h>
#include <Physicis/Collision/Collider/SphereCollider.h>
#include <Physicis/Collision/Collider/BoxCollider.h>
#include <Utility/HavokFeatures.h>

Scene::Scene(QObject* parent)
	: IScene(parent),
	  m_camera(new Camera(NULL,this)),
	  m_absoluteTime(0.0f),
	  m_relativeTime(0.0f),
	  m_delayedTime(0.0f),
	  m_bShowSkybox(false),
	  m_bPhysicsPaused(true),
	  m_bStepPhysics(false),
	  m_physicsWorld(0)
{
}

Scene::~Scene()
{
	clearScene();
	SAFE_DELETE(m_camera);
	SAFE_DELETE(m_sceneRootNode);
	m_havokPhysicsWorld->removeReference();
}

Scene* Scene::m_instance = 0;

Scene* Scene::instance()
{
	static QMutex mutex;
	if (!m_instance) 
	{
		QMutexLocker locker(&mutex);
		if (!m_instance)
			m_instance = new Scene;
	}

	return m_instance;
}

void Scene::initialize()
{
	Q_ASSERT(initializeOpenGLFunctions());

	initPhysicsModule();
	initHavokPhysicsModule();

	//glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glClearDepth( 1.0 );

	m_meshManager     = MeshManager::instance();
	m_textureManager  = TextureManager::instance();
	m_materialManager = MaterialManager::instance();
	m_objectManager   = ObjectManager::instance();
	
	m_sceneRootNode = new GameObject;
	m_sceneRootNode->setObjectName("Scene Root");

	resetToDefaultScene();
}

void Scene::initPhysicsModule()
{
	PhysicsWorldConfig config;

	m_physicsWorld = new PhysicsWorld(config);

}

void Scene::update(float currentTime)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// update the camera
	m_camera->update(currentTime - m_relativeTime);

	// record the absolute time
	m_absoluteTime = currentTime;

	// update the time
	float dt = m_absoluteTime - m_delayedTime - m_relativeTime;
	m_relativeTime = m_absoluteTime - m_delayedTime;

	// do nothing when the physics world is paused
	if (!m_bPhysicsPaused || m_bStepPhysics)
	{
		// update the physics world
		//m_physicsWorld->simulate(dt);
		m_havokPhysicsWorld->stepDeltaTime(dt);
	}
	m_physicsWorld->setCurrentTime(m_relativeTime);

	// render sky box
	if (m_bShowSkybox) 
		m_skybox->render(m_relativeTime);

	// render all
	m_objectManager->renderAll(m_relativeTime);

	// always reset the flag of step physics
	m_bStepPhysics = false;
}

Camera* Scene::getCamera()
{
	return m_camera;
}

void Scene::clearScene()
{
	m_materialManager->clear();
	m_textureManager->clear();
	m_meshManager->clear();
	m_objectManager->clear();
	m_lights.clear();

	emit ligthsChanged();
	emit updateHierarchy();
	emit cleared();

	// reset the physics world
	// put this line in the last, because
	// the destructor of a rigid body will automatically remove itself from the physics world
	// and the line below will also remove the rigid body list in the world
	m_physicsWorld->reset();
}

void Scene::resetToDefaultScene()
{
	glClearColor(0.39f, 0.39f, 0.39f, 0.0f);
	clearScene();
	toggleDebugMode(false);
	m_camera->resetCamera();
	m_currentSceneFile.clear();

	// Initializing the lights
	LightPtr light = createLight();
	light->gameObject()->setFixedPositionY(2);

	// back up
// 	GameObjectPtr temp(new GameObject(this));
// 	temp->setPosition(0, -1000, 0);
// 	temp->setScale(0.001);
// 	LoaderThread backup(this, "../Resource/Models/Common/sphere.obj", temp, m_sceneRootNode);

	// load the floor
	GameObjectPtr floorRef(new GameObject);
	LoaderThread floorLoader("../Resource/Models/Common/DemoRoom/WoodenFloor.obj", floorRef);
	GameObjectPtr floorObject = m_objectManager->getGameObject("WoodenFloor");
	ModelPtr floor = floorObject ->getComponent("Model").dynamicCast<IModel>();
	RigidBodyPtr floorBody = RigidBodyPtr(new RigidBody());
	floorBody->setMotionType(RigidBody::MOTION_FIXED);
	floorBody->attachBroadPhaseCollider(floor->getBoundingBox());
	floorBody->attachNarrowPhaseCollider(floor->getConvexHullCollider());
	floorObject->attachComponent(floorBody);
	m_physicsWorld->addEntity(floorBody.data());
}

void Scene::reloadScene()
{
	pause();
	if (!m_currentSceneFile.isEmpty())
		loadScene(m_currentSceneFile);
	else
		resetToDefaultScene();
}

void Scene::showLoadModelDialog()
{
	pause();

	QString fileName = QFileDialog::getOpenFileName(0, tr("Load Model"),
		"../Resource/Models",
		tr("3D Model File (*.dae *.obj *.3ds)"));

	LoaderThread loader(fileName, GameObjectPtr());
}

void Scene::showOpenSceneDialog()
{
	// pause the scene when the dialog shows up
	// to avoid frame drops
	pause();

	QString fileName = QFileDialog::getOpenFileName(0, tr("Open Scene"),
		"../Resource/Scenes",
		tr("Scene File (*.nebula)"));

	if (!fileName.isEmpty())
	{
		m_currentSceneFile = fileName;
		loadScene(fileName);
	}
}

void Scene::showSaveSceneDialog()
{
	// pause the scene when the dialog shows up
	// to avoid frame drops
	pause();

	QString fileName = QFileDialog::getSaveFileName(0, tr("Save Scene"),
		"../Resource/Scenes/scene",
		tr("Scene File (*.nebula)"));

	if (!fileName.isEmpty())
		saveScene(fileName);
}

void Scene::loadScene( const QString& fileName )
{
	QFile file(fileName);
	if (!file.open(QIODevice::ReadOnly))
	{
		qWarning() << "Unable to open file:" << fileName;
		return;
	}

	// clear the scene and process it
	clearScene();

	QDataStream in(&file);
	in.setVersion(QDataStream::Qt_5_4);

	in >> this;

	file.close();

	toggleDebugMode(false);

	qDebug() << "Opened scene from" << fileName;

	// make sure to send out this signal
	emit updateHierarchy();

	bridgeToHavokPhysicsWorld();
}

void Scene::saveScene( const QString& fileName )
{
	QFile file(fileName);
	if (!file.open(QIODevice::WriteOnly))
	{
		qWarning() << "Unable to save to file:" << fileName;
		return;
	}

	QDataStream out(&file);
	out.setVersion(QDataStream::Qt_5_4);

	out << this;

	file.flush();
	file.close();

	qDebug() << "Saved scene to" << fileName;

	m_currentSceneFile = fileName;
}

void Scene::loadPrefab( const QString& fileName )
{
	QFile file(fileName);
	if (!file.open(QIODevice::ReadOnly))
	{
		qWarning() << "Unable to open file:" << fileName;
		return;
	}

	QDataStream in(&file);
	in.setVersion(QDataStream::Qt_5_4);

	GameObjectPtr go = createEmptyGameObject();
	in >> go;

	// check if this object has the same name with another
	QString name = go->objectName();
	go->setPosition(0, 0, 0);
	go->setRotation(0, 0, 0);

	int duplication = 0;
	foreach(QString key, m_objectManager->m_gameObjectMap.keys())
	{
		if(key.contains(name)) 
			++duplication;
	}
	if (duplication) 
		name += "_" + QString::number(duplication + 1);

	m_objectManager->renameGameObject(go, name);

	file.close();

	// make sure to send out this signal
	emit updateHierarchy();
}

void Scene::savePrefab( const QString& fileName, const GameObjectPtr objectOut )
{
	QFile file(fileName);
	if (!file.open(QIODevice::WriteOnly))
	{
		qWarning() << "Unable to save to file:" << fileName;
		return;
	}

	QDataStream out(&file);
	out.setVersion(QDataStream::Qt_5_4);

	out << objectOut;

	file.flush();
	file.close();
}

void Scene::modelLoaded()
{
	emit updateHierarchy();
}

GameObjectPtr Scene::createEmptyGameObject(const QString& name, GameObject* parent)
{
	GameObject* ref = parent ? parent : m_sceneRootNode;

	GameObjectPtr go = m_objectManager->createGameObject(name, ref);

	emit updateHierarchy();
	return go;
}

ParticleSystemPtr Scene::createParticleSystem(GameObject* objectToAttach)
{
	GameObjectPtr ref = createEmptyGameObject("Particle System", objectToAttach);

	// particle system
	ParticleSystemPtr ps(new ParticleSystem);
	ref->attachComponent(ps);
	ps->initParticleSystem();
	ps->assingCollisionObject(m_objectManager->getGameObject("Floor"));

	return ps;
}

LightPtr Scene::createLight( GameObject* objectToAttach )
{
	GameObjectPtr ref = createEmptyGameObject("Light", objectToAttach);

	// light
	LightPtr l(new Light);
	addLight(l);
	ref->attachComponent(l);

	return l;
}

RigidBodyPtr Scene::createRigidBody( GameObject* objectToAttach )
{
	RigidBodyPtr rb;

	// check if the object already has a rigid body
	if (objectToAttach->getComponent("RigidBody"))
	{
		qWarning() << "This game object already has a Rigid Body Component. Operation ignored.";
		return rb;
	}

	// check if the object has a mesh reference
	ModelPtr model = objectToAttach->getComponent("Model").dynamicCast<IModel>();
	if (!model)
	{
		qWarning() << "A Rigid Body should have a mesh representation. Please load a model first.";
		return rb;
	}
	
	rb.reset(new RigidBody());
	rb->setTransform(objectToAttach->getTransform());
	
	// attach the broad and narrow phase collider(box and convex hull by default)
	rb->attachBroadPhaseCollider(model->getBoundingBox());
	rb->attachNarrowPhaseCollider(model->getConvexHullCollider());
	objectToAttach->attachComponent(rb);
	m_physicsWorld->addEntity(rb.data());

	return rb;
}

void Scene::setBackGroundColor( const QColor& col )
{
	glClearColor(col.redF(), col.greenF(), col.blueF(), 0.0f);
}

void Scene::toggleSkybox( bool state )
{
	m_bShowSkybox = state;
	
	if (state)
	{
		m_skybox.reset(new Skybox);
// 		m_skybox->init(
// 			"../Resource/Textures/skybox/Vasa/posx.jpg",
// 			"../Resource/Textures/skybox/Vasa/negx.jpg",
// 			"../Resource/Textures/skybox/Vasa/posy.jpg",
// 			"../Resource/Textures/skybox/Vasa/negy.jpg",
// 			"../Resource/Textures/skybox/Vasa/posz.jpg",
// 			"../Resource/Textures/skybox/Vasa/negz.jpg");
		m_skybox->init(
			"../Resource/Textures/skybox/Nebula/PurpleNebula2048_left.tif",
			"../Resource/Textures/skybox/Nebula/PurpleNebula2048_right.tif",
			"../Resource/Textures/skybox/Nebula/PurpleNebula2048_top.tif",
			"../Resource/Textures/skybox/Nebula/PurpleNebula2048_bottom.tif",
			"../Resource/Textures/skybox/Nebula/PurpleNebula2048_front.tif",
			"../Resource/Textures/skybox/Nebula/PurpleNebula2048_back.tif");
	}
	else
	{
		m_skybox.clear();
	}
}

void Scene::toggleAA(bool state)
{
	(state) ? glEnable(GL_MULTISAMPLE) : glDisable(GL_MULTISAMPLE);
}

void Scene::togglePoints(bool state)
{
	if(state)
	{
		glDisable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	}
}

void Scene::toggleWireframe(bool state)
{
	if(state)
	{
		glDisable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
}

void Scene::toggleFill(bool state)
{
	if(state)
	{
		glEnable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

void Scene::resize(int width, int height)
{
	glViewport(0, 0, width, height);

	if(m_camera->projectionType() == Camera::Perspective)
	{
		float aspect = static_cast<float>(width) / static_cast<float>(height);

		m_camera->setPerspectiveProjection(m_camera->fieldOfView(),
										   aspect,
										   m_camera->nearPlane(),
										   m_camera->farPlane());
	}
	else if(m_camera->projectionType() == Camera::Orthogonal)
	{
		m_camera->setOrthographicProjection(m_camera->left(),
											m_camera->right(),
											m_camera->bottom(),
											m_camera->top(),
											m_camera->nearPlane(),
											m_camera->farPlane());
	}
}

void Scene::pause()
{
	// pause the physics world
	m_bPhysicsPaused = true;
	m_physicsWorld->lock();

	m_canvas->getContainerWidget()->setWindowTitle("Scene - Physics Simulation: Off");
}

void Scene::play()
{
	// un-pause the physics world and set the delayed time
	m_bPhysicsPaused = false;
	m_physicsWorld->unlock();
	m_physicsWorld->start();
	m_delayedTime += m_absoluteTime - m_relativeTime;

	m_canvas->getContainerWidget()->setWindowTitle("Scene - Physics Simulation: On");
}

void Scene::step()
{
	// pause the physics world first
	m_bPhysicsPaused = true;
	m_physicsWorld->unlock();
	m_bStepPhysics = true;

	m_canvas->getContainerWidget()->setWindowTitle("Scene - Physics Simulation: Step");
}

void Scene::toggleDebugMode( bool state )
{
	if (state)
	{
		// attach all the bounding volumes
		foreach(ComponentPtr comp, m_objectManager->m_renderQueue)
		{
			ModelPtr model = comp.dynamicCast<IModel>();
			if (model && !model->getCurrentBoundingVolume()->gameObject())
				model->showBoundingVolume();
		}
	}
	else
	{
		// detach all the bounding volumes
		foreach(ComponentPtr comp, m_objectManager->m_renderQueue)
		{
			ModelPtr model = comp.dynamicCast<IModel>();
			if (model && model->getCurrentBoundingVolume()->gameObject())
				model->hideBoundingVolume();
		}
	}
}

void Scene::onLightChanged( Light* l )
{
	// find the light in the list
	LightPtr source;

	foreach(LightPtr light, m_lights)
	{
		if (light.data() == l)
		{
			source = light;
			break;
		}
	}
	
	// check if the source exists,
	// if it does, emit the signal
	if (source)
	{
		emit ligthsChanged();
	}
}

void Scene::removeLight( Light* l )
{
	foreach(LightPtr light, m_lights)
	{
		if (light.data() == l)
		{
			m_lights.removeOne(light);
			emit ligthsChanged();
			break;
		}
	}
}

void Scene::addLight( LightPtr l )
{
	m_lights << l;
	emit ligthsChanged();
}

void Scene::initHavokPhysicsModule()
{
	hkMemorySystem::FrameInfo finfo(500 * 1024);	// Allocate 500KB of Physics solver buffer
	hkMemoryRouter* memoryRouter = hkMemoryInitUtil::initDefault(hkMallocAllocator::m_defaultMallocAllocator, finfo);
	hkBaseSystem::init( memoryRouter, errorReport );

	// Create the physics world
	hkpWorldCinfo worldInfo;
	worldInfo.setupSolverInfo(hkpWorldCinfo::SOLVER_TYPE_4ITERS_MEDIUM);
	worldInfo.m_gravity = hkVector4(0.0f, -9.8f, 0.0f);
	worldInfo.m_broadPhaseBorderBehaviour = hkpWorldCinfo::BROADPHASE_BORDER_FIX_ENTITY; // just fix the entity if the object falls off too far

	// You must specify the size of the broad phase - objects should not be simulated outside this region
	worldInfo.setBroadPhaseWorldSize(1000.0f);
	m_havokPhysicsWorld = new hkpWorld(worldInfo);

	// Register all collision agents, even though only box - box will be used in this particular example.
	// It's important to register collision agents before adding any entities to the world.
	hkpAgentRegisterUtil::registerAllAgents( m_havokPhysicsWorld->getCollisionDispatcher() );
}

void Scene::bridgeToHavokPhysicsWorld()
{
	// reads all entities from native physics world
	m_havokPhysicsWorld->removeAll();
	QList<PhysicsWorldObject*> entityList = m_physicsWorld->getEntityList();
	foreach(PhysicsWorldObject* obj, entityList)
	{
		RigidBody* rb = (RigidBody*)obj;

		hkpRigidBodyCinfo bodyInfo;
		hkpRigidBody* hkRB = NULL;
		hkpShape* shape = NULL;

		RigidBody::MotionType motion = rb->getMotionType();
		switch(motion)
		{
		case RigidBody::MOTION_BOX_INERTIA:
			bodyInfo.m_motionType == hkpMotion::MOTION_BOX_INERTIA;
			break;
		case  RigidBody::MOTION_SPHERE_INERTIA:
			bodyInfo.m_motionType == hkpMotion::MOTION_SPHERE_INERTIA;
			break;
		case RigidBody::MOTION_FIXED:
			bodyInfo.m_motionType == hkpMotion::MOTION_FIXED;
			break;
		}

		ColliderPtr collider = rb->getBroadPhaseCollider();
		BoxColliderPtr box = collider.dynamicCast<BoxCollider>();
		SphereColliderPtr sphere = collider.dynamicCast<SphereCollider>();
		hkMassProperties massProperties;
		if (box)
		{
			hkVector4 boxSize = Math::Converter::toHkVec4(0.85f*box->getHalfExtents());
			shape = new hkpBoxShape(boxSize);

			// Compute mass properties
			hkpInertiaTensorComputer::computeBoxVolumeMassProperties(boxSize, rb->getMass(), massProperties);
		}
		if (sphere)
		{
			shape = new hkpSphereShape(sphere->getRadius());

			// Compute mass properties
			hkpInertiaTensorComputer::computeSphereVolumeMassProperties(sphere->getRadius(), rb->getMass(), massProperties);
		}

		bodyInfo.m_shape = shape;
		bodyInfo.m_inertiaTensor = massProperties.m_inertiaTensor;
		bodyInfo.m_centerOfMass = massProperties.m_centerOfMass;
		bodyInfo.m_mass = massProperties.m_mass;
		bodyInfo.m_position = Math::Converter::toHkVec4(rb->getPosition());
		bodyInfo.m_rotation = Math::Converter::toHkQuat(rb->getRotation());
		bodyInfo.m_linearVelocity = Math::Converter::toHkVec4(rb->getLinearVelocity());
		bodyInfo.m_angularVelocity = Math::Converter::toHkVec4(rb->getAngularVelocity());
		bodyInfo.m_restitution = rb->getRestitution();
		bodyInfo.m_gravityFactor = rb->getGravityFactor();
		
		hkRB = new hkpRigidBody(bodyInfo);
		m_havokPhysicsWorld->addEntity(hkRB);

		rb->setHkReference(hkRB);

		shape->removeReference();
		hkRB->removeReference();
	}

	// Create the floor as a fixed box
	{
		hkpRigidBodyCinfo boxInfo;
		hkVector4 boxSize(50.0f, 0.01f , 50.0f);
		hkpBoxShape* boxShape = new hkpBoxShape(boxSize);
		boxInfo.m_shape = boxShape;
		boxInfo.m_motionType = hkpMotion::MOTION_FIXED;
		boxInfo.m_position.set(0.0f, 0.0f, 0.0f);
		boxInfo.m_restitution = 0.9f;

		hkpRigidBody* floor = new hkpRigidBody(boxInfo);
		boxShape->removeReference();

		m_havokPhysicsWorld->addEntity(floor);
		floor->removeReference();
	}
}
