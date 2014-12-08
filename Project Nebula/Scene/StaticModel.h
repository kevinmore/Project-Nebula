#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLShaderProgram>
#include <Scene/AbstractModel.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Managers/MaterialManager.h>
#include <Scene/Managers/TextureManager.h>
#include <Primitives/Mesh.h>
#include <Primitives/Texture.h>
#include <Primitives/Material.h>
#include <Utility/ModelLoader.h>
#include <Scene/GameObject.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>

class Scene;

class StaticModel : public AbstractModel
{
public:
	StaticModel(Scene* scene, ShadingTechnique* tech, const GLuint vao);
	StaticModel(Scene* scene, ShadingTechnique* tech, const GLuint vao, QVector<ModelDataPtr> modelDataVector);
	virtual ~StaticModel(void);

	virtual void render( float time );
	bool hasAnimation() { return m_hasAnimation; }
	GameObject* getActor() { return m_actor; }
	ShadingTechnique* getShadingTech() { return m_RenderingEffect; }

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

	GLuint m_vao;

	QOpenGLFunctions_4_3_Core* m_funcs;
	Scene* m_scene;
	ShadingTechnique* m_RenderingEffect;
	bool m_hasAnimation;
	GameObject* m_actor;
};

