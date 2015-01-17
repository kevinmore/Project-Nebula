#include "Scene.h"
#include <Utility/LoaderThread.h>
#include <Utility/Serialization.h>

Scene::Scene(QObject* parent)
	: AbstractScene(parent),
	  m_sceneRootNode(new GameObject),
	  m_camera(new Camera(NULL,this)),
	  m_light("light01"),
	  m_lightMode(PerFragmentPhong),
	  m_lightModeSubroutines(LightModeCount),
	  m_time(0.0f)
{
	m_sceneRootNode->setObjectName("Scene Root");
	// Initializing the lights
	for(int i = 1; i < LightModeCount; ++i)
	{
		m_lightModeSubroutines[i] = i;
	}
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
	m_funcs = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_4_3_Core>();

	if( ! m_funcs )
	{
		qFatal("Requires OpenGL >= 4.3");
		exit(1);
	}

	m_funcs->initializeOpenGLFunctions();

	m_funcs->glClearDepth( 1.0 );
	m_funcs->glEnable(GL_DEPTH_TEST);
	m_funcs->glDepthFunc(GL_LEQUAL);
	m_funcs->glClearColor(0.39f, 0.39f, 0.39f, 0.0f);

	m_funcs->glEnable(GL_CULL_FACE);

//     m_light.setType(Light::SpotLight);
//     m_light.setUniqueColor(1.0, 1.0, 1.0);
//     m_light.setAttenuation(1.0f, 0.14f, 0.07f);
//     m_light.setIntensity(3.0f);

	m_objectManager = QSharedPointer<ObjectManager>(new ObjectManager(this));
	m_materialManager = QSharedPointer<MaterialManager>(new MaterialManager(1));// hack, fix it later!
	m_textureManager = QSharedPointer<TextureManager>(new TextureManager());
	m_meshManager = QSharedPointer<MeshManager>(new MeshManager());


	m_stateMachine = new QStateMachine();


	resetToDefaultScene();
}


void Scene::update(float currentTime)
{
	float dt = currentTime - m_time;
	m_time = currentTime;

	m_camera->update(dt);

	m_funcs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// render all
	m_objectManager->renderAll(currentTime);

// 	m_shaderProgram->bind();
// 	m_shaderProgram->setUniformValue("normalMatrix", normalMatrix);
// 	m_shaderProgram->setUniformValue("modelMatrix", m_object.modelMatrix());
// 	m_shaderProgram->setUniformValue("viewMatrix", m_camera->viewMatrix());
// 	m_shaderProgram->setUniformValue("projectionMatrix", m_camera->projectionMatrix());


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

void Scene::render(double currentTime)
{
	// Set the fragment shader light mode subroutine
//     m_funcs->glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &m_lightModeSubroutines[m_lightMode]);
// 
// 	if(currentTime > 0)
// 	{
// 		m_object.rotateY(static_cast<float>(currentTime)/0.02f);
// 	}

	emit renderCycleDone();
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

void Scene::toggleFill(bool state)
{
	if(state)
	{
		glEnable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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

void Scene::togglePoints(bool state)
{
	if(state)
	{
		glDisable(GL_CULL_FACE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	}
}

void Scene::togglePhong(bool state)
{
	if(state) m_lightMode = PerFragmentPhong;
}

void Scene::toggleBlinnPhong(bool state)
{
	if(state) m_lightMode = PerFragmentBlinnPhong;
}

void Scene::toggleRimLighting(bool state)
{
	if(state) m_lightMode = RimLighting;
}

void Scene::toggleAA(bool state)
{
	(state) ? glEnable(GL_MULTISAMPLE) : glDisable(GL_MULTISAMPLE);
}

Camera* Scene::getCamera()
{
	return m_camera;
}

QSharedPointer<MeshManager> Scene::meshManager()
{
	return m_meshManager;
}

QSharedPointer<TextureManager> Scene::textureManager()
{
	return m_textureManager;
}

QSharedPointer<MaterialManager> Scene::materialManager()
{
	return m_materialManager;
}

QSharedPointer<ObjectManager> Scene::modelManager()
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
	clearScene();
	m_camera->resetCamera();

	// load the floor
	GameObject* floorRef = new GameObject;
	floorRef->setRotation(-90.0f, 0.0f, 0.0f);
	LoaderThread loader(this, "../Resource/Models/DemoRoom/floor.DAE", floorRef, m_sceneRootNode);
}

void Scene::showLoadModelDialog()
{
	QString fileName = QFileDialog::getOpenFileName(0, tr("Load Model"),
		"../Resource/Models",
		tr("3D Model File (*.dae *.obj *.3ds)"));

	LoaderThread loader(this, fileName, 0, m_sceneRootNode);
}

void Scene::showOpenSceneDialog()
{
	QString fileName = QFileDialog::getOpenFileName(0, tr("Open Scene"),
		"../Resource/Scenes",
		tr("Scene File (*.nebula)"));

	loadScene(fileName);
}

void Scene::showSaveSceneDialog()
{
	QString fileName = QFileDialog::getSaveFileName(0, tr("Save Scene"),
		"../Resource/Scenes",
		tr("Scene File (*.nebula)"));

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

	QDataStream in(&file);
	in.setVersion(QDataStream::Qt_5_3);

	in >> this;

	// clear the scene and load the models
	clearScene();

	for (int i = 0; i < m_objectManager->m_modelsInfo.size(); ++i)
	{
		QString modelFileName = m_objectManager->m_modelsInfo[i].first;
		GameObject* go = m_objectManager->m_modelsInfo[i].second;

		LoaderThread loader(this, modelFileName, go, m_sceneRootNode);
	}

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

	m_objectManager->gatherModelsInfo();

	out << this;

	file.flush();
	file.close();

	qDebug() << "Saved scene to" << fileName;
}

void Scene::modelLoaded()
{
	emit updateHierarchy();
}

void Scene::createEmptyGameObject()
{
	m_objectManager->createGameObject("Game Object", m_sceneRootNode);

	emit updateHierarchy();
}

void Scene::createParticleSystem()
{
	GameObjectPtr ref = m_objectManager->createGameObject("Particle System", m_sceneRootNode);
	emit updateHierarchy();

	// particle system
	ParticleSystemPtr ps(new ParticleSystem(this));
	ref->attachComponent(ps);
	ps->initParticleSystem();
	ps->setGeneratorProperties(
		vec3(-20, 50, -20), // Minimal velocity
		vec3(20, 80, 20), // Maximal velocity
		vec3(0, -9.8, 0), // Gravity force applied to particles
		vec3(0.0f, 0.5f, 1.0f), // Color (light blue)
		30.0f, // Minimum lifetime in seconds
		40.0f, // Maximum lifetime in seconds
		0.75f, // Rendered size
		0.01f, // Spawn every 0.05 seconds
		50); // And spawn 30 particles

}
