#pragma once
#include <QtGui/QOpenGLFunctions_4_3_Core>
#include <QtGui/QOpenGLShaderProgram>
#include <Scene/AbstractModel.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Managers/MaterialManager.h>
#include <Scene/Managers/TextureManager.h>
#include <Primitives/Mesh.h>
#include <Primitives/Texture.h>
#include <Primitives/Material.h>
#include <Utility/ModelLoader.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/CCDIKSolver.h>
#include <Animation/IK/FABRIKSolver.h>

class Scene;

class RiggedModel : public AbstractModel
{
public:
	RiggedModel(const QString& name, Scene* scene, ShadingTechnique* tech, Skeleton* skeleton, FKController* fkCtrl, CCDIKSolver* ikSolver);
	RiggedModel(const QString& name, Scene* scene, ShadingTechnique* tech, Skeleton* skeleton, FKController* fkCtrl, CCDIKSolver* ikSolver, QVector<ModelDataPtr> modelDataVector);
	virtual ~RiggedModel();

	virtual void render( float time );

	bool hasAnimation() { return m_hasAnimation; }
	float animationDuration();
	void setActor(GameObject* actor) { m_actor = actor; }
	void setReachableTargetPos(vec3& pos);
	Skeleton* getSkeleton() { return m_skeleton; }
	void setRootTranslation(vec3& delta) { m_rootPositionTranslation = delta; }
	vec3 getRootTranslation() { return m_rootPositionTranslation; }
	void setRootRotation(QQuaternion& delta) {m_rootRotationTranslation = delta; }
	QQuaternion getRootRotation() { return m_rootRotationTranslation; }

protected:
	QVector<MeshPtr> m_meshes;
	QVector<QVector<TexturePtr>>  m_textures;
	QVector<MaterialPtr> m_materials;

	QSharedPointer<MeshManager>     m_meshManager;
	QSharedPointer<TextureManager>  m_textureManager;
	QSharedPointer<MaterialManager> m_materialManager;

private:
	enum DrawingMode
	{
		Indexed,
		Instanced,
		BaseVertex
	};

	void initRenderingEffect();
	void initialize(QVector<ModelDataPtr> modelDataVector = QVector<ModelDataPtr>());
	void drawElements(unsigned int index, int mode);
	void destroy();

	GLuint m_vao;

	QOpenGLFunctions_4_3_Core* m_funcs;
	Scene* m_scene;
	ShadingTechnique* m_RenderingEffect;
	bool m_hasAnimation;
	Skeleton* m_skeleton;
	FKController* m_FKController;
	CCDIKSolver* m_CCDSolver;
	FABRIKSolver* m_FABRSolver;
	bool ikSolved;
	float updateIKRate, lastUpdatedTime;
	vec3 m_targetPos;
	float solvingDuration;

	float m_animationDuration;
	vec3 m_rootPositionTranslation;
	QQuaternion m_rootRotationTranslation;
};

