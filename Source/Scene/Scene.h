#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <QSharedPointer>
#include <QOpenGLShaderProgram>
#include <Scene/AbstractScene.h>
#include <Scene/Light.h>
#include <Scene/Camera.h>
#include <Scene/GameObject.h>
#include <Scene/RiggedModel.h>
#include <Scene/Managers/TextureManager.h>
#include <Scene/Managers/MaterialManager.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Managers/ObjectManager.h>
#include <Animation/StateMachine/AnimatorController.h>
#include <Animation/StateMachine/NPCController.h>
#include <Physicis/Particles/ParticleSystem.h>

typedef QSharedPointer<QOpenGLShaderProgram> ShadersProgramPtr;

class Scene : public AbstractScene
{
	Q_OBJECT

public:
	Scene(QObject* parent = 0);
	virtual ~Scene();

	virtual void initialize();
	virtual void update(float currentTime);
	virtual void render(double currentTime);
	virtual void resize(int width, int height);

	enum LightMode
	{
		PerFragmentBlinnPhong = 1,
		PerFragmentPhong,
		RimLighting,
		LightModeCount
	};

	void setLightMode(LightMode lightMode) { m_lightMode = lightMode; }
	LightMode lightMode() const { return m_lightMode; }

	Camera* getCamera();
	
	QSharedPointer<MeshManager>     meshManager();
	QSharedPointer<TextureManager>  textureManager();
	QSharedPointer<MaterialManager> materialManager();
	QSharedPointer<ObjectManager>    modelManager();

	QStateMachine* getStateMachine() const { return m_stateMachine; }
	GameObject* sceneNode() const { return m_sceneRootNode; }

public slots:
	void toggleFill(bool state);
	void toggleWireframe(bool state);
	void togglePoints(bool state);

	void togglePhong(bool state);
	void toggleBlinnPhong(bool state);
	void toggleRimLighting(bool state);

	void toggleAA(bool state);
	void showLoadModelDialog();
	void resetToDefaultScene();
	void clearScene();
	void showOpenSceneDialog();
	void showSaveSceneDialog();
	void modelLoaded();
	void createEmptyGameObject();
	void createParticleSystem();

signals:
	void renderCycleDone();
	void updateHierarchy();

private:
	void loadScene(QString& fileName);
	void saveScene(QString& fileName);

	GameObject* m_sceneRootNode;

	Camera* m_camera;

	ModelPtr m_model;

	QSharedPointer<ObjectManager>    m_objectManager;
	QSharedPointer<MeshManager>     m_meshManager;
	QSharedPointer<TextureManager>  m_textureManager;
	QSharedPointer<MaterialManager> m_materialManager;

	ShadersProgramPtr m_shaderProgram;
	Light			  m_light;
	
	float m_time;

	LightMode       m_lightMode;
	QVector<GLuint> m_lightModeSubroutines;

	QOpenGLFunctions_4_3_Core* m_funcs;
	QVector<vec3> m_path;

	QStateMachine* m_stateMachine;
	AnimatorController* m_playerController;
	QVector<NPCController*> m_NPCs;
	ParticleSystem* m_particleSystem;
};

