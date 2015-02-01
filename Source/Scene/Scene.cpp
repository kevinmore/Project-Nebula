#include "Scene.h"
#include <Utility/LoaderThread.h>
#include <Utility/Serialization.h>
#include <Primitives/Puppet.h>
#include <Physicis/Entity/BoxRigidBody.h>
#include <Physicis/Entity/SphereRigidBody.h>

Scene::Scene(QObject* parent)
	: AbstractScene(parent),
	  m_camera(new Camera(NULL,this)),
	  m_lightMode(PerFragmentPhong),
	  m_absoluteTime(0.0f),
	  m_relativeTime(0.0f),
	  m_delayedTime(0.0f),
	  m_bShowSkybox(false),
	  m_bPaused(false),
	  m_physicsWorld(0)
{
	// Initializing the lights
	m_light = LightPtr(new Light("Main Light"));
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

	glClearDepth( 1.0 );
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glEnable(GL_CULL_FACE);

//     m_light.setType(Light::SpotLight);
//     m_light.setUniqueColor(1.0, 1.0, 1.0);
//     m_light.setAttenuation(1.0f, 0.14f, 0.07f);
//     m_light.setIntensity(3.0f);

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
// 	GameObjectPtr go = createEmptyGameObject("Rigid Cube");
// 	LoaderThread loader(this, "../Resource/Models/Common/cube.obj", go, m_sceneRootNode, false);
// 	BoxRigidBodyPtr cube(new BoxRigidBody());
// 	cube->setPosition(vec3(0, 1, 0));
// 	//cube->setLinearVelocity(vec3(50,0,0));
// 	cube->setAngularVelocity(vec3(0, 10, 0));
// 	cube->setMass(100);
// 	cube->setGravityFactor(0.0f);
// 	go->attachComponent(cube);
// 	m_physicsWorld->addEntity(cube.data());

	GameObjectPtr go = createEmptyGameObject("Rigid Sphere");
	LoaderThread loader(this, "../Resource/Models/Common/woodenball.obj", go, m_sceneRootNode, false);
	SphereRigidBodyPtr sphere(new SphereRigidBody());
	sphere->setPosition(vec3(0, 1, 0));
	//cube->setLinearVelocity(vec3(50,0,0));
	//sphere->setAngularVelocity(vec3(0, 10, 0));
	//sphere->setMass(100);
	sphere->setGravityFactor(0.0f);
	go->attachComponent(sphere);
	m_physicsWorld->addEntity(sphere.data());

	// particle system
// 	GameObjectPtr particle = createParticleSystem("Rigid Cube");
// 	particle->setPosition(0.45f, -0.50f, 0.45f);
// 	particle->setRotation(180, 0, 0);
// 	ComponentPtr comp = particle->getComponent("ParticleSystem");
// 	ParticleSystemPtr ps = comp.dynamicCast<ParticleSystem>();
// 	ps->setGravityFactor(0.1);
// 	ps->toggleRandomColor(true);
// 	ps->setMaxLife(4);
// 	ps->setMinLife(2);
// 	ps->toggleCollision(true);
// 	ps->setRestitution(0.2);
// 	ps->setMaxVelX(0);
// 	ps->setMaxVelZ(0);
// 	ps->setMinVelX(0);
// 	ps->setMinVelZ(0);

// 	particle = createParticleSystem("Rigid Cube");
// 	particle->setPosition(45, -50, -45);
// 	particle->setRotation(180, 0, 0);
// 	comp = particle->getComponent("ParticleSystem");
// 	ps = comp.dynamicCast<ParticleSystem>();
// 	ps->toggleRandomColor(true);
// 	ps->setMaxLife(4);
// 	ps->setMinLife(2);
// 	ps->toggleCollision(true);
// 	ps->setRestitution(0.2);
// 
// 	particle = createParticleSystem("Rigid Cube");
// 	particle->setPosition(-45, -50, 45);
// 	particle->setRotation(180, 0, 0);
// 	comp = particle->getComponent("ParticleSystem");
// 	ps = comp.dynamicCast<ParticleSystem>();
// 	ps->toggleRandomColor(true);
// 	ps->setMaxLife(4);
// 	ps->setMinLife(2);
// 	ps->toggleCollision(true);
// 	ps->setRestitution(0.2);
// 
// 	particle = createParticleSystem("Rigid Cube");
// 	particle->setPosition(-45, -50, -45);
// 	particle->setRotation(180, 0, 0);
// 	comp = particle->getComponent("ParticleSystem");
// 	ps = comp.dynamicCast<ParticleSystem>();
// 	ps->toggleRandomColor(true);
// 	ps->setMaxLife(4);
// 	ps->setMinLife(2);
// 	ps->toggleCollision(true);
// 	ps->setRestitution(0.2);


// 	GameObjectPtr go2 = createEmptyGameObject("Rigid Cube 0 Gravity");
// 	LoaderThread loader2(this, "../Resource/Models/Common/MetalCube.3ds", go2, m_sceneRootNode, false);
// 	BoxRigidBodyPtr cube2(new BoxRigidBody());
// 	cube2->setPosition(vec3(-100, 100, 0));
// 	cube2->setGravityFactor(0.0f);
// 	go2->attachComponent(cube2);
// 	m_physicsWorld->addEntity(cube2.data());
// 
// 	// particle system
// 	particle = createParticleSystem("Rigid Cube 0 Gravity");
// 	particle->setPosition(50, 0, 50);
// 	particle->setRotation(0, 0, 90);
// 	comp = particle->getComponent("ParticleSystem");
// 	ps = comp.dynamicCast<ParticleSystem>();
// 	ps->toggleRandomColor(true);
// 	ps->setMaxLife(4);
// 	ps->setMinLife(2);
// 	ps->toggleCollision(true);
// 	ps->setRestitution(0.2);
// 
// 	particle = createParticleSystem("Rigid Cube 0 Gravity");
// 	particle->setPosition(-50, 0, -50);
// 	particle->setRotation(0, 0, -90);
// 	comp = particle->getComponent("ParticleSystem");
// 	ps = comp.dynamicCast<ParticleSystem>();
// 	ps->toggleRandomColor(true);
// 	ps->setMaxLife(4);
// 	ps->setMinLife(2);
// 	ps->toggleCollision(true);
// 	ps->setRestitution(0.2);
}


void Scene::initPhysicsModule()
{
	PhysicsWorldConfig config;

	m_physicsWorld = new PhysicsWorld(config);
}

void Scene::update(float currentTime)
{
	// update the camera
	m_camera->update(currentTime - m_relativeTime);

	// record the absolute time
	m_absoluteTime = currentTime;

	// update the time
	float dt = m_absoluteTime - m_delayedTime - m_relativeTime;
	m_relativeTime = m_absoluteTime - m_delayedTime;

	// do nothing when the scene is paused
	// not fully implemented
	if (!m_bPaused)
	{
		// update the physics world
		m_physicsWorld->update(dt);
	}
	

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// render skybox first
	if (m_bShowSkybox) 
		m_skybox->render(m_relativeTime);

	// render all
	m_objectManager->renderAll(m_relativeTime);


// 	m_light.setPosition(m_camera->position());
// 	m_light.setDirection(m_camera->viewCenter());
// 	m_light.render(m_shaderProgram, m_camera->viewMatrix());

	
	// make the object to follow a curve path
// 	QSharedPointer<StaticModel> target = m_modelManager->getModel("coffecup").dynamicCast<StaticModel>();
// 	vec3 curPos = m_path[qFloor(t*500)%m_path.size()];
// 	target->getActor()->setPosition(curPos);

	// pass in the target position to the Rigged Model
// 	QSharedPointer<RiggedModel> man = m_modelManager->getModel("m005").dynamicCast<RiggedModel>();
// 	vec3 modelSpaceTargetPos = vec3(curPos.x(), -curPos.y(), -curPos.z()); // hack, need to multiply by several matrixes
// 	man->setReachableTargetPos(modelSpaceTargetPos);


	// render all static models
	//m_modelManager->renderStaticModels(t);

	
	//m_modelManager->renderRiggedModels(t);

	/*
	// render NPCs
	for (int i = 0; i < m_NPCs.size(); ++i)
	{
		m_NPCs[i]->render(t);
		// checking distance to the player
		QVector<NPCController*> socialTargets = m_playerController->getSocialTargets();
		float distance = (m_NPCs[i]->getActor()->position() - m_playerController->getActor()->position()).length();
		int index = socialTargets.indexOf(m_NPCs[i]);
		if (distance > 200.0f && index > -1)
			socialTargets.removeAt(index);
		if (distance <= 200.0f && index == -1)
			socialTargets.push_back(m_NPCs[i]);
	}
	*/

	// render the character controlled by the user
	//if(m_modelManager->m_riggedModels.size() > 0) m_playerController->render(t);

	
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

	emit updateHierarchy();
}

void Scene::resetToDefaultScene()
{
	glClearColor(0.39f, 0.39f, 0.39f, 0.0f);
	clearScene();
	m_camera->resetCamera();

	// load the floor
	GameObjectPtr floorRef(new GameObject(this));
	LoaderThread loader(this, "../Resource/Models/Common/DemoRoom/WoodenFloor.obj", floorRef, m_sceneRootNode);
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

	loadScene(fileName);

	play();
}

void Scene::showSaveSceneDialog()
{
	// pause the scene when the dialog shows up
	// to avoid frame drops
	pause();

	QString fileName = QFileDialog::getSaveFileName(0, tr("Save Scene"),
		"../Resource/Scenes",
		tr("Scene File (*.nebula)"));

	saveScene(fileName);

	play();
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
			"../Resource/Textures/skybox/interstellar_ft.tga",
			"../Resource/Textures/skybox/interstellar_bk.tga",
			"../Resource/Textures/skybox/interstellar_up.tga",
			"../Resource/Textures/skybox/interstellar_dn.tga",
			"../Resource/Textures/skybox/interstellar_rt.tga",
			"../Resource/Textures/skybox/interstellar_lf.tga");
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
	// pause the scene
	m_bPaused = true;
}

void Scene::play()
{
	// un-pause the scene and set the delayed time
	m_bPaused = false;
	m_delayedTime += m_absoluteTime - m_relativeTime;
}
