#include "Scene.h"


Scene::Scene(QObject* parent)
	: AbstractScene(parent),
	  m_camera(new SceneCamera(this)),
	  m_light("light01"),
	  m_v(),
	  m_viewCenterFixed(false),
	  m_panAngle(0.0f),
	  m_tiltAngle(0.0f),
	  m_time(0.0f),
	  m_metersToUnits(0.05f),
	  m_lightMode(PerFragmentPhong),
	  m_lightModeSubroutines(LightModeCount)
{
	// Initializing the position and orientation of the camera
	m_camera->setPosition(QVector3D(0.0f, 200.0f, 200.0f));
	m_camera->setViewCenter(QVector3D(0.0f, 100.0f, 0.0f));
	m_camera->setUpVector(Math::Vector3D::UNIT_Y);

	// Initializing the lights
	for(int i = 1; i < LightModeCount; ++i)
	{
		m_lightModeSubroutines[i] = i;
	}
}

Scene::~Scene()
{
	delete m_camera;
	m_camera = nullptr;
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


	m_modelManager->loadModel("floor", "../Resource/Models/DemoRoom/floor.DAE", ModelLoader::STATIC_MODEL);
	m_modelManager->loadModel("coffecup", "../Resource/Models/IK_Lab/coffecup.DAE", ModelLoader::STATIC_MODEL);
	m_modelManager->loadModel("m005", "../Resource/Models/IK_Lab/m005.DAE", ModelLoader::RIGGED_MODEL);

	// generate a bezier curve
	QVector<vec3> anchors;
// 	anchors << vec3(-150, 100, 0) << vec3(-50, 120, 150) << vec3(0, 150, 100) << vec3(50, 80, 20)
// 		<< vec3(100, 0, -50) << vec3(150, 80, -100) << vec3(150, 200, -120) << vec3(100, 180, -90)
// 		<< vec3(50, 100, -50) << vec3(0, 50, -20) << vec3(-100, 80, -10) << vec3(-150, 100, 0);
	anchors << vec3(0, 0, 100) << vec3(20, 20, 80) << vec3(40, 40, 60) << vec3(60, 60, 40) << vec3(40, 80, 20)
			<< vec3(20, 100, -20) << vec3(0, 120, -40) << vec3(-20, 140, -60) << vec3(-40, 160, -80)
			<< vec3(-60, 180, -100) << vec3(-40, 200, -80) << vec3(-20, 220, -60) << vec3(0, 240, -40)
			<< vec3(20, 220, -20) << vec3(40, 200, 0) << vec3(60, 180, 20) << vec3(40, 160, 40)
			<< vec3(20, 140, 60) << vec3(0, 120, 80) << vec3(-20, 100, 100) << vec3(-40, 80, 80)
			<< vec3(-60, 60, 60) << vec3(-40, 40, 80) << vec3(-20, 20, 90) << vec3(0, 0, 100); 
	m_BezierPath = Math::Spline::makeBezier3D(anchors);
}


void Scene::update(float t)
{
	const float dt = t - m_time;
	m_time = t;

	SceneCamera::CameraTranslationOption option = m_viewCenterFixed
		? SceneCamera::DontTranslateViewCenter
		: SceneCamera::TranslateViewCenter;

	m_camera->translate(m_v * dt * m_metersToUnits, option);

	if( ! qFuzzyIsNull(m_panAngle) )
	{
		m_camera->pan(m_panAngle, QVector3D(0.0f, 1.0f, 0.0f));
		m_panAngle = 0.0f;
	}

	if ( ! qFuzzyIsNull(m_tiltAngle) )
	{
		m_camera->tilt(m_tiltAngle);
		m_tiltAngle = 0.0f;
	}

	m_funcs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


// 	m_shaderProgram->bind();
// 	m_shaderProgram->setUniformValue("normalMatrix", normalMatrix);
// 	m_shaderProgram->setUniformValue("modelMatrix", m_object.modelMatrix());
// 	m_shaderProgram->setUniformValue("viewMatrix", m_camera->viewMatrix());
// 	m_shaderProgram->setUniformValue("projectionMatrix", m_camera->projectionMatrix());


// 	m_light.setPosition(m_camera->position());
// 	m_light.setDirection(m_camera->viewCenter());
// 	m_light.render(m_shaderProgram, m_camera->viewMatrix());

	// make the floor look nicer
// 	QSharedPointer<StaticModel> floorMesh = m_modelManager->getModel("floor").dynamicCast<StaticModel>();
// 	floorMesh->getShadingTech()->Enable();
// 	floorMesh->getShadingTech()->SetMatSpecularIntensity(100.f);
// 	floorMesh->getShadingTech()->SetMatSpecularPower(46.0f);
// 	floorMesh->getShadingTech()->Disable();
	
	// make the object to follow a curve path
	QSharedPointer<StaticModel> target = m_modelManager->getModel("coffecup").dynamicCast<StaticModel>();
	vec3 curPos = m_BezierPath[qFloor(t*1000)%m_BezierPath.size()];
	target->getActor()->setPosition(curPos);

	// pass in the target position to the Rigged Model
	QSharedPointer<RiggedModel> man = m_modelManager->getModel("m005").dynamicCast<RiggedModel>();
	vec3 modelSpaceTargetPos = vec3(curPos.x(), -curPos.y(), -curPos.z()); // hack, need to multiply by several matrixes
	man->setReachableTargetPos(modelSpaceTargetPos);

	m_modelManager->renderAllModels(t);
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

	if(m_camera->projectionType() == SceneCamera::PerspectiveProjection)
	{
		float aspect = static_cast<float>(width) / static_cast<float>(height);

		m_camera->setPerspectiveProjection(m_camera->fieldOfView(),
										   aspect,
										   m_camera->nearPlane(),
										   m_camera->farPlane());
	}
	else if(m_camera->projectionType() == SceneCamera::OrthogonalProjection)
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

SceneCamera* Scene::getCamera()
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







