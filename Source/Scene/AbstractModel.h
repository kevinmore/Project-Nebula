#pragma once
#include <QString>
#include <QSharedPointer>
#include <QVector4D>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>

#include <Scene/GameObject.h>
#include <Primitives/Component.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>

struct MeshData
{
	QString name;

	unsigned int numIndices;
	unsigned int baseVertex;
	unsigned int baseIndex;
};


struct TextureData
{
	QString colorMap, normalMap;
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
	float animationDuration;
};

typedef QSharedPointer<MeshData> MeshDataPtr;
typedef QSharedPointer<TextureData> TextureDataPtr;
typedef QSharedPointer<MaterialData> MaterialDataPtr;
typedef QSharedPointer<ModelData> ModelDataPtr;

class AbstractModel : public Component
{
public:
	AbstractModel(ShadingTechniquePtr tech = ShadingTechniquePtr(), const QString& fileName = "");
	virtual ~AbstractModel() = 0;

	virtual void render(const float currentTime) = 0;
	virtual QString className() { return "Model"; }

	QString fileName() const { return m_fileName; }
	void setFileName(QString& file) { m_fileName = file; }

	ShadingTechniquePtr renderingEffect() const { return m_RenderingEffect; }

protected:
	QString m_fileName;
	ShadingTechniquePtr m_RenderingEffect;
};