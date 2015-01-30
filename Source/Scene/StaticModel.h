#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <Scene/AbstractModel.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Managers/MaterialManager.h>
#include <Scene/Managers/TextureManager.h>
#include <Utility/ModelLoader.h>

class Scene;

class StaticModel : public AbstractModel, protected QOpenGLFunctions_4_3_Core
{
public:
	StaticModel(const QString& name, Scene* scene, ShadingTechniquePtr tech);
	StaticModel(const QString& name, Scene* scene, ShadingTechniquePtr tech, QVector<ModelDataPtr> modelDataVector);
	
	// copy constructor
	StaticModel::StaticModel( const StaticModel* orignal );

	virtual ~StaticModel();

	virtual void render( const float currentTime );
	//virtual QString className() { return "StaticModel"; }
	ShadingTechniquePtr getShadingTech() const { return m_RenderingEffect; }

	Scene* getScene() const { return m_scene; }
	QVector<ModelDataPtr> getModelData() const { return m_modelDataVector; }

protected:

	MeshManager*     m_meshManager;
	TextureManager*  m_textureManager;
	MaterialManager* m_materialManager;

private:
	enum DrawingMode
	{
		Indexed,
		Instanced,
		BaseVertex
	};

	void initialize(QVector<ModelDataPtr> modelDataVector = QVector<ModelDataPtr>());
	void drawElements(unsigned int index, int mode);

	Scene* m_scene;
	QVector<ModelDataPtr> m_modelDataVector;
};

