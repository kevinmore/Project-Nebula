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
	m_camera->setPosition(QVector3D(-8.0f, 0.0f, -7.0f));
	m_camera->setViewCenter(QVector3D(0.0f, 0.0f, 0.0f));
	m_camera->setUpVector(QVector3D(0.0f, 1.0f, 0.0f));

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

	glShadeModel(GL_SMOOTH);
	glClearDepth( 1.0 );
    glClearColor(0.39f, 0.39f, 0.39f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_CULL_FACE);
	
	// compile and link shaders
	prepareShaders();
	m_shaderProgram->bind();
	m_shaderProgram->setUniformValue("texColor", 0);
	// shader->setUniformValue("texNormal", 1);
	
    m_light.setType(Light::SpotLight);
    m_light.setUniqueColor(1.0, 1.0, 1.0);
    m_light.setAttenuation(1.0f, 0.14f, 0.07f);
    m_light.setIntensity(3.0f);

	m_modelManager = QSharedPointer<ModelManager>(new ModelManager(this));
	m_materialManager = QSharedPointer<MaterialManager>(new MaterialManager(m_shaderProgram->programId()));
	m_textureManager = QSharedPointer<TextureManager>(new TextureManager());
	m_meshManager = QSharedPointer<MeshManager>(new MeshManager());

	
	m_model = m_modelManager->loadModel("Alice", "../Resource/Models/Alice/Alice.dae", m_shaderProgram);
}

void Scene::prepareShaders()
{
	m_shaderProgram = ShadersProgramPtr(new QOpenGLShaderProgram()); 
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, "../Resource/Shaders/basic.vert");
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, "../Resource/Shaders/basic.frag");
	m_shaderProgram->link();
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
}

void Scene::render(double currentTime)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	if(currentTime > 0)
	{
		m_object.rotateY(static_cast<float>(currentTime)/0.02f);
	}

	QMatrix4x4 modelViewMatrix = m_camera->viewMatrix() * m_object.modelMatrix();
	QMatrix3x3 normalMatrix = modelViewMatrix.normalMatrix();

	m_shaderProgram->bind();
	m_shaderProgram->setUniformValue("normalMatrix", normalMatrix);
	m_shaderProgram->setUniformValue("modelViewMatrix", modelViewMatrix);
	m_shaderProgram->setUniformValue("projectionMatrix", m_camera->projectionMatrix());

	m_model->render();

	m_light.setPosition(m_camera->position());
	m_light.setDirection(m_camera->viewCenter());
	m_light.render(m_shaderProgram, m_camera->viewMatrix());

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

Object3D* Scene::getObject()
{
	return &m_object;
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







