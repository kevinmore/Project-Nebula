#pragma once
#include <QtGui/QOpenGLFunctions_4_3_Core>
#include <QtCore/QSharedPointer>
#include <QtGui/QOpenGLShaderProgram>
#include <Scene/AbstractScene.h>
#include <Scene/Light.h>
#include <Scene/SceneCamera.h>
#include <Scene/Object3D.h>
#include <Scene/Model.h>
#include <Scene/Managers/TextureManager.h>
#include <Scene/Managers/MaterialManager.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Managers/ModelManager.h>
#include <Scene/ShadingTechniques/skinning_technique.h>

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

	// camera movement
	inline void setSideSpeed(float vx)     { m_v.setX(vx); }
	inline void setVerticalSpeed(float vy) { m_v.setY(vy); }
	inline void setForwardSpeed(float vz)  { m_v.setZ(vz); }
	inline void setViewCenterFixed(bool b) { m_viewCenterFixed = b; }

	// camera movement rotation
	inline void pan(float angle)  { m_panAngle  = angle; }
	inline void tilt(float angle) { m_tiltAngle = angle; }

	enum LightMode
	{
		PerFragmentBlinnPhong = 1,
		PerFragmentPhong,
		RimLighting,
		LightModeCount
	};

	void setLightMode(LightMode lightMode) { m_lightMode = lightMode; }
	LightMode lightMode() const { return m_lightMode; }

	Object3D*    getObject();
	SceneCamera* getCamera();

	QSharedPointer<MeshManager>     meshManager();
	QSharedPointer<TextureManager>  textureManager();
	QSharedPointer<MaterialManager> materialManager();

private:
	void prepareShaders();

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
	Object3D		  m_object;
	Light			  m_light;
	QVector3D		  m_v;

	bool m_viewCenterFixed;

	float m_panAngle;
	float m_tiltAngle;
	float m_time;

	const float m_metersToUnits;

	LightMode       m_lightMode;
	QVector<GLuint> m_lightModeSubroutines;

	QOpenGLFunctions_4_3_Core* m_funcs;
	GLuint m_boneLocation[200];

	SkinningTechnique* m_pEffect;
	DirectionalLight m_directionalLight;
};

