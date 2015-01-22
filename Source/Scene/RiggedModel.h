#pragma once
#include <QtGui/QOpenGLFunctions_4_3_Core>
#include <Scene/AbstractModel.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Managers/MaterialManager.h>
#include <Scene/Managers/TextureManager.h>
#include <Utility/ModelLoader.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/CCDIKSolver.h>
#include <Animation/IK/FABRIKSolver.h>

class Scene;

class RiggedModel : public AbstractModel, protected QOpenGLFunctions_4_3_Core
{
public:
	RiggedModel(const QString& name, Scene* scene, ShadingTechniquePtr tech, Skeleton* skeleton);
	RiggedModel(const QString& name, Scene* scene, ShadingTechniquePtr tech, Skeleton* skeleton, QVector<ModelDataPtr> modelDataVector);
	virtual ~RiggedModel();

	virtual QString className() { return "RiggedModel"; }
	virtual void render( const float currentTime );

	void setFKController(FKController* fkCtrl);
	void setIKSolver(CCDIKSolver* ikSolver);

	bool hasAnimation() const { return m_hasAnimation; }
	float animationDuration() const { return m_animationDuration; }

	void setActor(GameObject* actor) { m_actor = actor; }
	void setReachableTargetPos(vec3& pos);

	Skeleton* getSkeleton() const { return m_skeleton; }
	void setRootTranslation(vec3& delta) { m_rootPositionTranslation = delta; }
	vec3 getRootTranslation() const { return m_rootPositionTranslation; }
	void setRootRotation(QQuaternion& delta) {m_rootRotationTranslation = delta; }
	QQuaternion getRootRotation() const { return m_rootRotationTranslation; }

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

	Scene* m_scene;
	ShadingTechniquePtr m_RenderingEffect;
	bool m_hasAnimation;
	Skeleton* m_skeleton;
	FKController* m_FKController;
	CCDIKSolver* m_IKSolver;
	FABRIKSolver* m_FABRSolver;
	bool ikSolved;
	float updateIKRate, lastUpdatedTime;
	vec3 m_targetPos;
	float solvingDuration;

	float m_animationDuration;
	vec3 m_rootPositionTranslation;
	QQuaternion m_rootRotationTranslation;
};

