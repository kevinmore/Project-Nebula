#pragma once
#include <Scene/AbstractModel.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Managers/MaterialManager.h>
#include <Scene/Managers/TextureManager.h>
#include <Utility/ModelLoader.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/CCDIKSolver.h>
#include <Animation/IK/FABRIKSolver.h>

class Scene;

class RiggedModel : public AbstractModel
{
public:
	RiggedModel(const QString& name, Scene* scene, ModelLoaderPtr loader);
	RiggedModel(const QString& name, Scene* scene, ModelLoaderPtr loader, QVector<ModelDataPtr> modelDataVector);

	// copy constructor
	RiggedModel( const RiggedModel* orignal );

	virtual ~RiggedModel();

	//virtual QString className() { return "RiggedModel"; }
	virtual void render( const float currentTime );

	ShadingTechniquePtr getShadingTech() const { return m_renderingEffect; }

	Scene* getScene() const { return m_scene; }
	QVector<ModelDataPtr> getModelData() const { return m_modelDataVector; }
	ModelLoaderPtr getLoader() const { return m_modelLoader; }

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

	MeshManager*     m_meshManager;
	TextureManager*  m_textureManager;
	MaterialManager* m_materialManager;

private:

	void initialize(QVector<ModelDataPtr> modelDataVector = QVector<ModelDataPtr>());


	Scene* m_scene;
	QVector<ModelDataPtr> m_modelDataVector;
	ModelLoaderPtr m_modelLoader;

	bool m_hasAnimation;
	Skeleton* m_skeleton;
	FKController* m_FKController;
	CCDIKSolver* m_IKSolver;
	bool ikSolved;
	float updateIKRate, lastUpdatedTime;
	vec3 m_targetPos;
	float solvingDuration;

	float m_animationDuration;
	vec3 m_rootPositionTranslation;
	QQuaternion m_rootRotationTranslation;
};

