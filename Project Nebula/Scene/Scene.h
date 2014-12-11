#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <QSharedPointer>
#include <QOpenGLShaderProgram>
#include <Scene/AbstractScene.h>
#include <Scene/Light.h>
#include <Scene/SceneCamera.h>
#include <Scene/GameObject.h>
#include <Scene/RiggedModel.h>
#include <Scene/Managers/TextureManager.h>
#include <Scene/Managers/MaterialManager.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Managers/ModelManager.h>
#include <Animation/AnimatorController.h>

typedef QSharedPointer<QOpenGLShaderProgram> ShadersProgramPtr;

class Scene : public AbstractScene
{
	Q_OBJECT

public:
	Scene(QObject* parent = 0);
	virtual ~Scene();

	virtual void initialize();
	virtual void update(float t);
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

	GameObject*    getObject() { return NULL; } //hack! need to improve
	SceneCamera* getCamera();
	
	QSharedPointer<MeshManager>     meshManager();
	QSharedPointer<TextureManager>  textureManager();
	QSharedPointer<MaterialManager> materialManager();

	QStateMachine* getStateMachine() { return m_stateMachine; }


public slots:
	void toggleFill(bool state);
	void toggleWireframe(bool state);
	void togglePoints(bool state);

	void togglePhong(bool state);
	void toggleBlinnPhong(bool state);
	void toggleRimLighting(bool state);

	void toggleAA(bool state);

signals:
	void renderCycleDone();

private:
	SceneCamera* m_camera;

	ModelPtr m_model;

	QSharedPointer<ModelManager>    m_modelManager;
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
	AnimatorController* m_animCtrller;
	QVector<RiggedModel*> m_NPCs;
};

