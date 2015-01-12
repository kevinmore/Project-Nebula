#include "Scene.h"
#include <Utility/LoaderThread.h>
#include <Utility/Serialization.h>

Scene::Scene(QObject* parent)
	: AbstractScene(parent),
	  m_sceneNode(new GameObject),
	  m_camera(new Camera(NULL,this)),
	  m_light("light01"),
	  m_lightMode(PerFragmentPhong),
	  m_lightModeSubroutines(LightModeCount)
{
	m_sceneNode->setObjectName("Scene Root");
	// Initializing the lights
	for(int i = 1; i < LightModeCount; ++i)
	{
		m_lightModeSubroutines[i] = i;
	}
}

Scene::~Scene()
{
	SAFE_DELETE(m_camera);
	SAFE_DELETE(m_stateMachine);
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

	m_modelManager = QSharedPointer<ModelManager>(new ModelManager(this));
	m_materialManager = QSharedPointer<MaterialManager>(new MaterialManager(1));// hack, fix it later!
	m_textureManager = QSharedPointer<TextureManager>(new TextureManager());
	m_meshManager = QSharedPointer<MeshManager>(new MeshManager());

	resetToDefaultScene();


	/*
	// locomotions
 	m_modelManager->loadModel("m_idle", "../Resource/Models/Final/m005/m_idle.DAE");
 	m_modelManager->loadModel("m_run", "../Resource/Models/Final/m005/m_run.DAE");
 	m_modelManager->loadModel("m_run_start", "../Resource/Models/Final/m005/m_run_start.DAE");
 	m_modelManager->loadModel("m_run_stop", "../Resource/Models/Final/m005/m_run_stop.DAE");
 	m_modelManager->loadModel("m_run_to_walk", "../Resource/Models/Final/m005/m_run_to_walk.DAE");
 	m_modelManager->loadModel("m_turn_left_60_to_walk", "../Resource/Models/Final/m005/m_turn_left_60_to_walk.DAE");
 	m_modelManager->loadModel("m_turn_left_180_to_walk", "../Resource/Models/Final/m005/m_turn_left_180_to_walk.DAE");
 	m_modelManager->loadModel("m_turn_right_60_to_walk", "../Resource/Models/Final/m005/m_turn_right_60_to_walk.DAE");
	m_modelManager->loadModel("m_walk", "../Resource/Models/Final/m005/m_walk.DAE");
 	m_modelManager->loadModel("m_walk_start", "../Resource/Models/Final/m005/m_walk_start.DAE");
 	m_modelManager->loadModel("m_walk_stop", "../Resource/Models/Final/m005/m_walk_stop.DAE");
	m_modelManager->loadModel("m_walk_to_run", "../Resource/Models/Final/m005/m_walk_to_run.DAE");
	m_modelManager->loadModel("m_wave", "../Resource/Models/Final/m005/m_wave.DAE");
	m_modelManager->loadModel("m_talk", "../Resource/Models/Final/m005/m_talk.DAE");
 	m_modelManager->loadModel("m_listen", "../Resource/Models/Final/m005/m_listen.DAE");

	// set up the animator controller
	if(m_modelManager->m_riggedModels.size() == 0) return;
	m_playerController = new AnimatorController(m_modelManager);
	

	m_stateMachine = new QStateMachine();

	// NPCs
// 	m_modelManager->loadModel("f004_touch_hair", "../Resource/Models/Final/NPC/f004_touch_hair.DAE");
// 	m_modelManager->loadModel("f004_wave", "../Resource/Models/Final/NPC/f004_wave.DAE");
// 	NPCController* npc1 = new NPCController(m_modelManager, "f004_touch_hair", "f004_wave");
// 	npc1->getActor()->setPosition(0, 0, -600);
// 	m_NPCs << npc1;
// 	m_playerController->addSocialTargets(npc1);
// 
// 	m_modelManager->loadModel("f013_waiting", "../Resource/Models/Final/NPC/f013_waiting.DAE");
// 	m_modelManager->loadModel("f013_wave", "../Resource/Models/Final/NPC/f013_wave.DAE");
// 	NPCController* npc2 = new NPCController(m_modelManager, "f013_waiting", "f013_wave");
// 	npc2->getActor()->setPosition(-600, 0, 100);
// 	npc2->getActor()->setObjectYRotation(120);
// 	m_NPCs << npc2;
// 	m_playerController->addSocialTargets(npc2);
// 
// 	m_modelManager->loadModel("dog_idle", "../Resource/Models/Final/NPC/dog_idle.DAE");
// 	m_modelManager->loadModel("dog_bark", "../Resource/Models/Final/NPC/dog_bark.DAE");
// 	NPCController* npc3 = new NPCController(m_modelManager, "dog_idle", "dog_bark");
// 	npc3->getActor()->setPosition(600, 0, 100);
// 	npc3->getActor()->setObjectYRotation(-120);
// 	m_NPCs << npc3;
// 	m_playerController->addSocialTargets(npc3);

	m_playerController->buildStateMachine();
	m_stateMachine = m_playerController->getStateMachine();
	*/
	m_stateMachine = new QStateMachine();


	// generate a bezier curve
// 	QVector<vec3> anchors;
// 	anchors << vec3(-150, 100, 0) << vec3(-50, 120, 150) << vec3(0, 150, 100) << vec3(50, 80, 20) << vec3(80, 380, 0)
// 		<< vec3(100, 0, -50) << vec3(150, 80, -100) << vec3(0, 0, 0) << vec3(150, 400, -120) << vec3(100, 380, -90) << vec3(80, 250, -70)
// 		<< vec3(50, 100, -50) << vec3(0, 50, -20) << vec3(0, 0, 0) << vec3(-100, 80, -10) << vec3(-100, 0, 200) << vec3(-150, 100, 0);

	//m_path = Math::Spline::makeBezier3D(anchors);
	//m_path = Math::Spline::makeCatMullRomSpline(anchors);

// 	StaticModel* sceneObject = m_modelManager->getStaticModel("temple");
// 	sceneObject->getActor()->setScale(0.5);
// 	sceneObject->getActor()->setRotation(-90.0f, 0.0f, 0.0f);
// 	sceneObject->getActor()->setPosition(-100, 50, 1500);

// 	sceneObject = m_modelManager->getStaticModel("mountain");
// 	sceneObject->getActor()->setScale(0.5);
// 	sceneObject->getActor()->setRotation(-90.0f, 0.0f, 180.0f);
// 	sceneObject->getActor()->setPosition(-80, 250, 1100);
// 	sceneObject = m_modelManager->getStaticModel("floor");
// 	sceneObject->getActor()->setRotation(-90.0f, 0.0f, 0.0f);
	
// 	sceneObject = m_modelManager->getStaticModel("trolley");
// 	sceneObject->getActor()->setRotation(-90.0f, 0.0f, 0.0f);
// 	sceneObject->getActor()->setPosition(100.0f, 0, 0);
// 
// 	sceneObject = m_modelManager->getStaticModel("brick");
// 	sceneObject->getActor()->setPosition(-100.0f, 0, 0);
// 
// 	sceneObject = m_modelManager->getStaticModel("brick2");
// 	sceneObject->getActor()->setPosition(-100.0f, 100, 0);

	//m_camera->followTarget(m_playerController->getActor());

}


void Scene::update(float t)
{
	m_camera->update(t);

	m_funcs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


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

	// render all rigged models
	m_modelManager->renderAllModels(t);
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

QSharedPointer<ModelManager> Scene::modelManager()
{
	return m_modelManager;
}

void Scene::showLoadModelDialog()
{
	QString fileName = QFileDialog::getOpenFileName(0, tr("Load Model"),
		"../Resource/Models",
		tr("3D Model File (*.dae *.obj *.3ds)"));

	LoaderThread loader(this, fileName, 0, m_sceneNode);
	loader.run();
}

void Scene::clearScene()
{
	m_materialManager->clear();
	m_textureManager->clear();
	m_meshManager->clear();
	m_modelManager->clear();

	emit updateHierarchy();
}

void Scene::resetToDefaultScene()
{
	clearScene();
	m_camera->resetCamera();

	// load the floor
	m_modelManager->loadModel("floor", "../Resource/Models/DemoRoom/floor.DAE", m_sceneNode);
	StaticModel* sceneObject = m_modelManager->getStaticModel("floor");
	sceneObject->gameObject()->setRotation(-90.0f, 0.0f, 0.0f);

	emit updateHierarchy();
}

void Scene::showSaveSceneDialog()
{
	QString fileName = QFileDialog::getSaveFileName(0, tr("Save Scene"),
		"../Resource/Scenes",
		tr("Scene File (*.nebula)"));

	QFile file(fileName);
	if (!file.open(QIODevice::WriteOnly))
	{
		qWarning() << "Unable to save to file:" << fileName;
		return;
	}
	
	QDataStream out(&file);
	out.setVersion(QDataStream::Qt_5_3);

	m_modelManager->gatherModelsInfo();

	out << this;

	file.flush();
	file.close();
}

void Scene::showOpenSceneDialog()
{
	QString fileName = QFileDialog::getOpenFileName(0, tr("Open Scene"),
		"../Resource/Scenes",
		tr("Scene File (*.nebula)"));

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

	for (int i = 0; i < m_modelManager->m_modelsInfo.size(); ++i)
	{
		QString modelFileName = m_modelManager->m_modelsInfo[i].first;
		GameObject* go = m_modelManager->m_modelsInfo[i].second;

		LoaderThread loader(this, modelFileName, go, m_sceneNode);
 		loader.run();
	}

	file.close();
}

void Scene::modelLoaded()
{
	emit updateHierarchy();
}
