#pragma once
#include <QtGui/QOpenGLFunctions_4_3_Core>
#include <Scene/AbstractModel.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Managers/MaterialManager.h>
#include <Scene/Managers/TextureManager.h>
#include <Primitives/Mesh.h>
#include <Primitives/Texture.h>
#include <Primitives/Material.h>
#include <Utility/ModelLoader.h>

class Scene;

class Model : public AbstractModel
{
public:
	Model(Scene* scene, ModelLoader* loader, const QOpenGLVertexArrayObjectPtr& vao);
	Model(Scene* scene, ModelLoader* loader, const QOpenGLVertexArrayObjectPtr& vao,	QVector<ModelDataPtr> modelDataVector);
	virtual ~Model(void);

	virtual void render();
	bool hasAnimation() { return m_hasAnimation; }
	ModelLoader* m_loader;

protected:
	Scene* m_scene;
	bool m_hasAnimation;
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

	void initialize(QVector<ModelDataPtr> modelDataVector = QVector<ModelDataPtr>());
	void drawElements(unsigned int index, int mode);
	void destroy();

	QOpenGLVertexArrayObjectPtr m_vao;
	QOpenGLFunctions_4_3_Core *m_funcs;
};

