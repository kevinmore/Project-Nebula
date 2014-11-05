#pragma once
#include <QtCore/QString>
#include <QtCore/QSharedPointer>
#include <QtGui/QVector4D>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtGui/QOpenGLShaderProgram>

struct MeshData
{
	QString name;

	unsigned int numIndices;
	unsigned int baseVertex;
	unsigned int baseIndex;
};


struct TextureData
{
	QString filename;
	bool hasTexture;
};

struct MaterialData
{
	QString name;

	QVector4D ambientColor;
	QVector4D diffuseColor;
	QVector4D specularColor;
	QVector4D emissiveColor;

	float shininess;
	float shininessStrength;

	int twoSided;
	int blendMode;
	bool alphaBlending;
};

struct ModelData
{
	MeshData     meshData;
	TextureData  textureData;
	MaterialData materialData;
	bool hasAnimation;
};

typedef QSharedPointer<MeshData> MeshDataPtr;
typedef QSharedPointer<TextureData> TextureDataPtr;
typedef QSharedPointer<MaterialData> MaterialDataPtr;
typedef QSharedPointer<ModelData> ModelDataPtr;


typedef QSharedPointer<QOpenGLShaderProgram> QOpenGLShaderProgramPtr;
typedef QSharedPointer<QOpenGLVertexArrayObject> QOpenGLVertexArrayObjectPtr;

class AbstractModel
{


public:
	AbstractModel(Scene* scene, const QOpenGLVertexArrayObjectPtr vao);
	AbstractModel(Scene* scene, const QOpenGLVertexArrayObjectPtr vao, QVector<ModelDataPtr> modelDataVector);
	virtual ~AbstractModel();

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
	SkinningTechnique* m_RenderingEffect;
	bool m_hasAnimation;
	Object3D* m_actor;
	FKController* m_FKController;
	IKSolver* m_IKSolver;
};