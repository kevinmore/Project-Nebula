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
#include <Scene/Skybox.h>

#include <Animation/StateMachine/AnimatorController.h>
#include <Animation/StateMachine/NPCController.h>

#include <Physicis/Particles/ParticleSystem.h>
#include <Physicis/World/PhysicsWorld.h>

typedef QSharedPointer<QOpenGLShaderProgram> ShadersProgramPtr;

class Scene : public AbstractScene, protected QOpenGLFunctions_4_3_Core
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
	
	MeshManager*     meshManager();
	TextureManager*  textureManager();
	MaterialManager* materialManager();
	ObjectManager*  objectManager();

	QStateMachine* getStateMachine() const { return m_stateMachine; }
	GameObject* sceneNode() const { return m_sceneRootNode; }
	void setBackGroundColor(const QColor& col);
	SkyboxPtr getSkybox() const  { return m_skybox; }

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
	GameObjectPtr createEmptyGameObject(const QString& name = "Game Object");
	GameObjectPtr createParticleSystem(const QString& parentName = "Scene Root");

	void toggleSkybox(bool state);

	void pause();
	void play();


signals:
	void renderCycleDone();
	void updateHierarchy();

private:
	void loadScene(QString& fileName);
	void saveScene(QString& fileName);

	void initPhysicsModule();

	bool m_bPaused;
	

	GameObject* m_sceneRootNode;

	Camera* m_camera;

	ModelPtr m_model;

	ObjectManager*  m_objectManager;
	MeshManager*     m_meshManager;
	TextureManager* m_textureManager;
	MaterialManager* m_materialManager;

	ShadersProgramPtr m_shaderProgram;
	Light			  m_light;
	
	float m_absoluteTime; // absolute time from the start of the program
	float m_relativeTime; // relative time excluding paused duration 
	float m_delayedTime; // the time delayed between pauses

	LightMode       m_lightMode;
	QVector<GLuint> m_lightModeSubroutines;

	QVector<vec3> m_path;

	QStateMachine* m_stateMachine;
	AnimatorController* m_playerController;
	QVector<NPCController*> m_NPCs;
	ParticleSystem* m_particleSystem;
	SkyboxPtr m_skybox;
	bool m_bShowSkybox;
	//
	// Physics
	//
	PhysicsWorld* m_physicsWorld;
};

