#include "Scene.h"
#include <UI/Canvas.h>
#include <Utility/LoaderThread.h>
#include <Utility/Serialization.h>
#include <Primitives/Puppet.h>
#include <Physicis/Collision/Collider/SphereCollider.h>
#include <Physicis/Collision/Collider/BoxCollider.h>


//
// Forward declarations
//

void setupPhysics(hkpWorld* physicsWorld);
hkVisualDebugger* setupVisualDebugger(hkpPhysicsContext* worlds);
void stepVisualDebugger(hkVisualDebugger* vdb);
hkpRigidBody* g_ball;




























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
	SAFE_DELETE(m_stateMachine);
	SAFE_DELETE(m_sceneRootNode);
}

void Scene::initialize()
{
	Q_ASSERT(initializeOpenGLFunctions());

	initPhysicsModule();

	//glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glClearDepth( 1.0 );


	m_objectManager = new ObjectManager(this, this);
	m_materialManager = new MaterialManager(this);
	m_textureManager = new TextureManager(this);
	m_meshManager = new MeshManager(this);


	m_stateMachine = new QStateMachine();

	m_sceneRootNode = new GameObject(this);
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
		m_physicsWorld->simulate(dt);
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

MeshManager* Scene::meshManager()
{
	return m_meshManager;
}

TextureManager* Scene::textureManager()
{
	return m_textureManager;
}

MaterialManager* Scene::materialManager()
{
	return m_materialManager;
}

ObjectManager* Scene::objectManager()
{
	return m_objectManager;
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
	GameObjectPtr floorRef(new GameObject(this));
	LoaderThread floorLoader(this, "../Resource/Models/Common/DemoRoom/WoodenFloor.obj", floorRef, m_sceneRootNode);
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

	LoaderThread loader(this, fileName, GameObjectPtr(), m_sceneRootNode);
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
		"../Resource/Scenes",
		tr("Scene File (*.nebula)"));

	if (!fileName.isEmpty())
		saveScene(fileName);
}

void Scene::loadScene( QString& fileName )
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
	in.setVersion(QDataStream::Qt_5_3);

	in >> this;

	file.close();

	toggleDebugMode(false);

	qDebug() << "Opened scene from" << fileName;

	// make sure to send out this signal
	emit updateHierarchy();


	// broad phase demo
// 	{
// 		GameObjectPtr go = createEmptyGameObject("Boarder");
// 		go->setPosition(0, 6.1, 0);
// 		BoxColliderPtr boarder(new BoxCollider(vec3(0, 0, 0), vec3(6, 6, 6), this));
// 		boarder->setColor(Qt::cyan);
// 		go->attachComponent(boarder);
// 
// 		// create rigid bodies for simulation
// 		for(int i = 0; i < 50; ++i)
// 		{
// 			float ratio = Math::Random::random(0.5f, 1.2f);
// 			go = createEmptyGameObject("Alien");
// 			LoaderThread loader(this, "../Resource/Models/static/alien.obj", go, m_sceneRootNode, false);
// 			go->setPosition(Math::Random::random(vec3(-4, 1, -4), vec3(4, 9, 4)));
// 			RigidBodyPtr rb = createRigidBody(go.data());
// 			rb->setGravityFactor(0.0f);
// 			rb->setLinearVelocity(Math::Random::random(vec3(-4, -4, -4), vec3(4, 4, 4)));
// 			rb->setAngularVelocity(Math::Random::random(vec3(-1, -1, -1), vec3(1, 1, 1)));
// 			rb->setMotionType_SLOT("Sphere");
// 		}
// 	}
}

void Scene::saveScene( QString& fileName )
{
	QFile file(fileName);
	if (!file.open(QIODevice::WriteOnly))
	{
		qWarning() << "Unable to save to file:" << fileName;
		return;
	}

	QDataStream out(&file);
	out.setVersion(QDataStream::Qt_5_3);

	out << this;

	file.flush();
	file.close();

	qDebug() << "Saved scene to" << fileName;

	m_currentSceneFile = fileName;
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
	ParticleSystemPtr ps(new ParticleSystem(this));
	ref->attachComponent(ps);
	ps->initParticleSystem();
	ps->assingCollisionObject(m_objectManager->getGameObject("Floor"));

	return ps;
}

LightPtr Scene::createLight( GameObject* objectToAttach )
{
	GameObjectPtr ref = createEmptyGameObject("Light", objectToAttach);

	// light
	LightPtr l(new Light(this, ref.data()));
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
		m_skybox.reset(new Skybox(this));
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




// Keycode
#include <Common/Base/keycode.cxx>

// This excludes libraries that are not going to be linked
// from the project configuration, even if the keycodes are
// present
#undef HK_FEATURE_PRODUCT_AI
#undef HK_FEATURE_PRODUCT_ANIMATION
#undef HK_FEATURE_PRODUCT_CLOTH
#undef HK_FEATURE_PRODUCT_DESTRUCTION_2012
#undef HK_FEATURE_PRODUCT_DESTRUCTION
#undef HK_FEATURE_PRODUCT_BEHAVIOR
#undef HK_FEATURE_PRODUCT_MILSIM
#undef HK_FEATURE_PRODUCT_PHYSICS

#define HK_EXCLUDE_LIBRARY_hkpVehicle
#define HK_EXCLUDE_LIBRARY_hkCompat
#define HK_EXCLUDE_LIBRARY_hkSceneData
#define HK_EXCLUDE_LIBRARY_hkcdCollide

//
// Common
//
#define HK_EXCLUDE_FEATURE_SerializeDeprecatedPre700
#define HK_EXCLUDE_FEATURE_RegisterVersionPatches 
//#define HK_EXCLUDE_FEATURE_MemoryTracker

//
// Physics
//
#define HK_EXCLUDE_FEATURE_hkpHeightField
//#define HK_EXCLUDE_FEATURE_hkpSimulation
//#define HK_EXCLUDE_FEATURE_hkpContinuousSimulation
//#define HK_EXCLUDE_FEATURE_hkpMultiThreadedSimulation

#define HK_EXCLUDE_FEATURE_hkpAccurateInertiaTensorComputer

#define HK_EXCLUDE_FEATURE_hkpUtilities
#define HK_EXCLUDE_FEATURE_hkpVehicle
#define HK_EXCLUDE_FEATURE_hkpCompressedMeshShape
#define HK_EXCLUDE_FEATURE_hkpConvexPieceMeshShape
#define HK_EXCLUDE_FEATURE_hkpExtendedMeshShape
#define HK_EXCLUDE_FEATURE_hkpMeshShape
#define HK_EXCLUDE_FEATURE_hkpSimpleMeshShape
#define HK_EXCLUDE_FEATURE_hkpPoweredChainData
#define HK_EXCLUDE_FEATURE_hkMonitorStream

#include <Common/Base/Config/hkProductFeatures.cxx>

// Platform specific initialization
#include <Common/Base/System/Init/PlatformInit.cxx>