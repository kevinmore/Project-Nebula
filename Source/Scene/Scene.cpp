#include "Scene.h"
#include <Utility/LoaderThread.h>
#include <Utility/Serialization.h>
#include <Primitives/Puppet.h>
#include <Physicis/Collision/Collider/SphereCollider.h>
#include <Physicis/Collision/Collider/BoxCollider.h>

Scene::Scene(QObject* parent)
	: IScene(parent),
	  m_camera(new Camera(NULL,this)),
	  m_lightMode(PerFragmentPhong),
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

	// show sky box for demo purpose
	//toggleSkybox(true);

	// setup a basic physics world
	// setup a demo room

	// broad phase demo
// 	{
// 	 	GameObjectPtr go = createEmptyGameObject("Boarder");
// 	 	go->setPosition(0, 6, 0);
// 	 	BoxColliderPtr boarder(new BoxCollider(vec3(0, 6, 0), vec3(6, 6, 6), this));
// 	 	boarder->setColor(Qt::cyan);
// 	 	go->attachComponent(boarder);
// 	 
// 	 	// create rigid bodies for simulation
// 	 	for(int i = 0; i < 30; ++i)
// 	 	{
// 	 		float ratio = Math::Random::random(0.5f, 1.2f);
// 	 		go = createEmptyGameObject("Gundam");
// 	 		go->setScale(50);
// 	 		LoaderThread loader(this, "../Resource/Models/BroadPhaseDemo/robot.obj", go, m_sceneRootNode, false);
// 	 		SphereRigidBodyPtr rb(new SphereRigidBody());
// 	 		rb->setPosition(Math::Random::random(vec3(-4, 1, -4), vec3(4, 9, 4)));
// 	 		rb->setGravityFactor(0.0f);
// 	 		rb->setLinearVelocity(Math::Random::random(vec3(-4, -4, -4), vec3(4, 4, 4)));
// 	 		rb->setAngularVelocity(Math::Random::random(vec3(0, 0, 0), vec3(100, 100, 100)));
// 	 		go->attachComponent(rb);
// 	 		SphereColliderPtr collider(new SphereCollider(rb->getPosition(), 0.5f, this));
// 	     // BoxColliderPtr collider(new BoxCollider(rb->getPosition(), vec3(0.5, 0.5, 0.3), this));
// 	  		rb->attachCollider(collider);
// 	 		m_physicsWorld->addEntity(rb.data());
// 	 	}
// 	}



// 	BoxColliderPtr collider(new BoxCollider(vec3(0, 1, 0), vec3(0.5, 0.5, 0.5), this));
// 	go->attachComponent(collider);

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
	m_physicsWorld->reset();
}

void Scene::resetToDefaultScene()
{
	glClearColor(0.39f, 0.39f, 0.39f, 0.0f);
	clearScene();
	toggleDebugMode(false);
	m_camera->resetCamera();

	// Initializing the lights
	GameObjectPtr lightObject = createLight();
	lightObject->setFixedPositionY(2);

	// back up
	GameObjectPtr temp(new GameObject(this));
	temp->setPosition(0, -1000, 0);
	temp->setScale(0.001);
	LoaderThread backup(this, "../Resource/Models/Common/sphere.obj", temp, m_sceneRootNode);

	// load the floor
	GameObjectPtr floorRef(new GameObject(this));
	LoaderThread floorLoader(this, "../Resource/Models/Common/DemoRoom/WoodenFloor.obj", floorRef, m_sceneRootNode);
	GameObjectPtr floorObject = m_objectManager->getGameObject("WoodenFloor");
	ModelPtr floor = floorObject ->getComponent("Model").dynamicCast<IModel>();
	BoxRigidBodyPtr floorBody = BoxRigidBodyPtr(new BoxRigidBody());
	floor->getConvexHullCollider()->setRigidBody(floorBody.data());
	floorBody->setMotionType(RigidBody::MOTION_FIXED);
	floorBody->attachCollider(floor->getBoundingBox());
	floorObject->attachComponent(floorBody);
	m_physicsWorld->addEntity(floorBody.data());
// 
// 	GameObjectPtr go = createEmptyGameObject("Cube1");
// 	go->setRotation(45, 45, 0);
// 	LoaderThread loader(this, "../Resource/Models/Common/woodenbox.obj", go, m_sceneRootNode, false);
// 	BoxRigidBodyPtr rb(new BoxRigidBody());
// 	rb->setPosition(vec3(-1, 1, 0));
// 	rb->setGravityFactor(0.0f);
// 	rb->setLinearVelocity(vec3(1, 0, 0));
// 	ModelPtr model = m_objectManager->getGameObject("Cube1")->getComponent("Model").dynamicCast<IModel>();
// 	rb->attachCollider(model->getBoundingBox());
// 	model->getConvexHullCollider()->setRigidBody(rb.data());
// 	go->attachComponent(rb);
// 	m_physicsWorld->addEntity(rb.data());
// 
// 	GameObjectPtr go2 = createEmptyGameObject("Cube2");
// 	LoaderThread loader2(this, "../Resource/Models/Common/woodenbox.obj", go2, m_sceneRootNode, false);
// 	//LoaderThread loader2(this, "../Resource/Models/static/dragonTextured.obj", go2, m_sceneRootNode, false);
// 	BoxRigidBodyPtr rb2(new BoxRigidBody());
// 	rb2->setPosition(vec3(1, 1, 0));
// 	rb2->setGravityFactor(0.0f);
// 	rb2->setLinearVelocity(vec3(-1, 0, 0));
// 	rb2->setAngularVelocity(vec3(0, 1, 0));
// 	ModelPtr model2 = m_objectManager->getGameObject("Cube2")->getComponent("Model").dynamicCast<IModel>();
// 	rb2->attachCollider(model2->getBoundingBox());
// 	model2->getConvexHullCollider()->setRigidBody(rb2.data());
// 	go2->attachComponent(rb2);
// 	m_physicsWorld->addEntity(rb2.data());


	GameObjectPtr go2 = createEmptyGameObject("Cube2");
	LoaderThread loader2(this, "../Resource/Models/Common/woodenbox.obj", go2, m_sceneRootNode, false);
	BoxRigidBodyPtr rb2(new BoxRigidBody());
	rb2->setPosition(vec3(0, 1, 0));
	//rb2->setGravityFactor(0.0f);
	//rb2->setAngularVelocity(vec3(0, 1, 0));
	//rb2->applyAngularImpulse(vec3(0, 1 ,0));
	//rb2->applyPointImpulse(vec3(0, 0, -1), vec3(0.5, 1.5, 0.5));
	ModelPtr model2 = m_objectManager->getGameObject("Cube2")->getComponent("Model").dynamicCast<IModel>();
	rb2->attachCollider(model2->getBoundingBox());
	model2->getConvexHullCollider()->setRigidBody(rb2.data());
	go2->attachComponent(rb2);
	m_physicsWorld->addEntity(rb2.data());
}

void Scene::showLoadModelDialog()
{
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
		loadScene(fileName);
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

	qDebug() << "Opened scene from" << fileName;
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
}

void Scene::modelLoaded()
{
	emit updateHierarchy();
}

GameObjectPtr Scene::createEmptyGameObject(const QString& name)
{
	GameObjectPtr go = m_objectManager->createGameObject(name, m_sceneRootNode);

	emit updateHierarchy();
	return go;
}

GameObjectPtr Scene::createParticleSystem(const QString& parentName)
{
	GameObjectPtr parent = m_objectManager->getGameObject(parentName);
	// check if the parent exist
	GameObjectPtr ref = parent 
						? m_objectManager->createGameObject("Particle System", parent.data())
						: m_objectManager->createGameObject("Particle System", m_sceneRootNode);
	emit updateHierarchy();

	// particle system
	ParticleSystemPtr ps(new ParticleSystem(this));
	ref->attachComponent(ps);
	ps->initParticleSystem();
	ps->assingCollisionObject(m_objectManager->getGameObject("hexagonFloor"));

	return ref;
}

GameObjectPtr Scene::createLight( GameObject* parent )
{
	// check if the parent exist
	GameObjectPtr ref = parent 
		? m_objectManager->createGameObject("Light", parent)
		: m_objectManager->createGameObject("Light", m_sceneRootNode);

	// light
	LightPtr l(new Light(this, ref.data()));
	addLight(l);
	ref->attachComponent(l);

	emit updateHierarchy();

	return ref;
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
		m_skybox = SkyboxPtr(new Skybox(this));
		m_skybox->init(
			"../Resource/Textures/skybox/Vasa/posx.jpg",
			"../Resource/Textures/skybox/Vasa/negx.jpg",
			"../Resource/Textures/skybox/Vasa/posy.jpg",
			"../Resource/Textures/skybox/Vasa/negy.jpg",
			"../Resource/Textures/skybox/Vasa/posz.jpg",
			"../Resource/Textures/skybox/Vasa/negz.jpg");
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

void Scene::toggleRimLighting(bool state)
{
	if(state) m_lightMode = RimLighting;
}

void Scene::toggleBlinnPhong(bool state)
{
	if(state) m_lightMode = PerFragmentBlinnPhong;
}

void Scene::togglePhong(bool state)
{
	if(state) m_lightMode = PerFragmentPhong;
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
}

void Scene::play()
{
	// un-pause the physics world and set the delayed time
	m_bPhysicsPaused = false;
	m_physicsWorld->unlock();
	m_delayedTime += m_absoluteTime - m_relativeTime;
}

void Scene::step()
{
	// pause the physics world first
	m_bPhysicsPaused = true;
	m_physicsWorld->unlock();
	m_bStepPhysics = true;
}

void Scene::toggleDebugMode( bool state )
{
	if (state)
	{
		// attach all the bounding boxes
		foreach(ComponentPtr comp, m_objectManager->m_renderQueue)
		{
			ModelPtr model = comp.dynamicCast<IModel>();
			if (model && !model->getBoundingBox()->gameObject())
				model->showBoundingBox();
		}
	}
	else
	{
		// detach all the bounding boxes
		foreach(ComponentPtr comp, m_objectManager->m_renderQueue)
		{
			ModelPtr model = comp.dynamicCast<IModel>();
			if (model && model->getBoundingBox()->gameObject())
				model->hideBoundingBox();
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
