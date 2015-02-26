#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <QSharedPointer>
#include <QOpenGLShaderProgram>
#include <Scene/IScene.h>
#include <Scene/Light.h>
#include <Scene/Camera.h>
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
#include <Physicis/Entity/RigidBody.h>

typedef QSharedPointer<QOpenGLShaderProgram> ShadersProgramPtr;

class Scene : public IScene, protected QOpenGLFunctions_4_3_Core
{
	Q_OBJECT

public:
	Scene(QObject* parent = 0);
	virtual ~Scene();

	virtual void initialize();
	virtual void update(float currentTime);
	virtual void resize(int width, int height);

	Camera* getCamera();
	
	MeshManager*     meshManager();
	TextureManager*  textureManager();
	MaterialManager* materialManager();
	ObjectManager*  objectManager();

	QStateMachine* getStateMachine() const { return m_stateMachine; }
	GameObject* sceneRoot() const { return m_sceneRootNode; }
	void setBackGroundColor(const QColor& col);
	SkyboxPtr getSkybox() const  { return m_skybox; }
	QList<LightPtr> getLights() const { return m_lights; }
	void removeLight(Light* l);
	void addLight(LightPtr l);
	bool isSkyBoxEnabled() const { return m_bShowSkybox; }

public slots:
	void toggleFill(bool state);
	void toggleWireframe(bool state);
	void togglePoints(bool state);

	void toggleAA(bool state);
	void showLoadModelDialog();
	void resetToDefaultScene();
	void clearScene();
	void showOpenSceneDialog();
	void showSaveSceneDialog();
	void modelLoaded();
	GameObjectPtr createEmptyGameObject(const QString& name = "Game Object", GameObject* parent = 0);
	GameObjectPtr createParticleSystem(GameObject* parent = 0);
	GameObjectPtr createLight(GameObject* parent = 0);
	RigidBodyPtr createRigidBody(GameObject* objectToAttach);

	void toggleSkybox(bool state);

	void toggleDebugMode(bool state);
	void pause();
	void play();
	void step();

	void onLightChanged(Light* l);

signals:
	void updateHierarchy();
	void ligthsChanged();
	void cleared();

private:
	void loadScene(QString& fileName);
	void saveScene(QString& fileName);

	void initPhysicsModule();

	mutable bool m_bPhysicsPaused, m_bStepPhysics;
	
	GameObject* m_sceneRootNode;

	Camera* m_camera;

	ModelPtr m_model;

	ObjectManager*  m_objectManager;
	MeshManager*     m_meshManager;
	TextureManager* m_textureManager;
	MaterialManager* m_materialManager;

	QList<LightPtr>		  m_lights;
	
	float m_absoluteTime; // absolute time from the start of the program
	float m_relativeTime; // relative time excluding paused duration 
	float m_delayedTime; // the time delayed between pauses

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

