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
#include <Scene/Object3D.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>
#include <Animation/FK/FKController.h>
#include <Animation/IK/IKSolver.h>

class Scene;

class RiggedModel : public AbstractModel
{
public:
	RiggedModel(Scene* scene, FKController* fkCtrl, IKSolver* ikSolver, const QOpenGLVertexArrayObjectPtr vao);
	RiggedModel(Scene* scene, FKController* fkCtrl, IKSolver* ikSolver, const QOpenGLVertexArrayObjectPtr vao, QVector<ModelDataPtr> modelDataVector);
	virtual ~RiggedModel(void);

	virtual void render( float time );
	bool hasAnimation() { return m_hasAnimation; }
	Object3D* getActor() { return m_actor; }

protected:
	QVector<MeshPtr> m_meshes;
	QVector<TexturePtr>  m_textures;
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

	QOpenGLVertexArrayObjectPtr m_vao;

	QOpenGLFunctions_4_3_Core* m_funcs;
	Scene* m_scene;
	ShadingTechnique* m_RenderingEffect;
	bool m_hasAnimation;
	Object3D* m_actor;
	FKController* m_FKController;
	IKSolver* m_IKSolver;

};

