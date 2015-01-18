#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLShaderProgram>
#include <Scene/AbstractModel.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Managers/MaterialManager.h>
#include <Scene/Managers/TextureManager.h>
#include <Utility/ModelLoader.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>

class Scene;

class StaticModel : public AbstractModel
{
public:
	StaticModel(const QString& name, Scene* scene, ShadingTechnique* tech);
	StaticModel(const QString& name, Scene* scene, ShadingTechnique* tech, QVector<ModelDataPtr> modelDataVector);
	virtual ~StaticModel();

	virtual void render( const float currentTime );
	virtual QString className() { return "StaticModel"; }
	ShadingTechnique* getShadingTech() { return m_RenderingEffect; }

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
};

