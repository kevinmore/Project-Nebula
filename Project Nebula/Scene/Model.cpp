#include "Model.h"
#include <Scene/Scene.h>
#include <QtGui/QOpenGLContext>

Model::Model(Scene* scene, const QOpenGLVertexArrayObjectPtr& vao)
  : m_scene(scene),
	m_vao(vao),
	m_funcs(nullptr)
{
	initialize();
}

Model::Model(Scene* scene, const QOpenGLVertexArrayObjectPtr& vao, QVector<ModelDataPtr> modelData)
  : m_scene(scene),
	m_vao(vao),
	m_funcs(nullptr)
{
	initialize(modelData);
}

Model::~Model() {}

void Model::initialize(QVector<ModelDataPtr> modelDataVector)
{
	QOpenGLContext* context = QOpenGLContext::currentContext();

	Q_ASSERT(context);

	m_funcs = context->versionFunctions<QOpenGLFunctions_4_3_Core>();
	m_funcs->initializeOpenGLFunctions();

	m_meshManager     = m_scene->meshManager();
	m_textureManager  = m_scene->textureManager();
	m_materialManager = m_scene->materialManager();

	// traverse modelData vector
	for (int i = 0; i < modelDataVector.size(); ++i)
	{
		ModelDataPtr data = modelDataVector[i];

		// deal with the mesh
		MeshPtr mesh = m_meshManager->getMesh(data->meshData.name);
		if (!mesh)
		{
			mesh = m_meshManager->addMesh(data->meshData.name, data->meshData.numIndices, data->meshData.baseVertex, 	data->meshData.baseIndex);
		}

		m_meshes.push_back(mesh);

		// deal with the texture
		if(data->textureData.hasTexture)
		{
			TexturePtr  texture = m_textureManager->getTexture(data->textureData.filename);

			if(!texture)
			{
				texture = m_textureManager->addTexture(data->textureData.filename, data->textureData.filename);
			}

			m_textures.push_back(texture);
		}
		else m_textures.push_back(TexturePtr(nullptr));

		// deal with the material
		MaterialPtr material = m_materialManager->getMaterial(data->materialData.name);
		if(!material)
		{
			material = m_materialManager->addMaterial(data->materialData.name,
													data->materialData.ambientColor,
													data->materialData.diffuseColor,
													data->materialData.specularColor,
													data->materialData.emissiveColor,
													data->materialData.shininess,
													data->materialData.shininessStrength,
													data->materialData.twoSided,
													data->materialData.blendMode,
													data->materialData.alphaBlending,
													data->textureData.hasTexture);
		}

		m_materials.push_back(material);

	}
	
}

void Model::destroy() {}

void Model::render()
{
	m_vao->bind();

	for(int i = 0; i < m_meshes.size(); ++i)
	{
		if( m_materials[i] != nullptr && ! m_materials[i]->isTranslucent())
		{
			if(m_textures[i] != nullptr)
			{
				m_textures[i]->bind(GL_TEXTURE0);
			}

			m_materials[i]->bind();

			drawElements(i, Indexed | BaseVertex);
		}
	}

	for(int i = 0; i < m_meshes.size(); ++i)
	{
		if( m_materials[i] != nullptr && m_materials[i]->isTranslucent())
		{
			glDepthMask(GL_FALSE);
			glEnable(GL_BLEND);

			m_materials[i]->bind();

			drawElements(i, Indexed | BaseVertex);

			glDisable(GL_BLEND);
			glDepthMask(GL_TRUE);
		}
	}

	m_vao->release();
}

void Model::drawElements(unsigned int index, int mode)
{
	// Mode has not been implemented yet
	Q_UNUSED(mode);

	m_funcs->glDrawElementsBaseVertex(
		GL_TRIANGLES,
		m_meshes[index]->getNumIndices(),
		GL_UNSIGNED_INT,
		reinterpret_cast<void*>((sizeof(unsigned int)) * m_meshes[index]->getBaseIndex()),
		m_meshes[index]->getBaseVertex()
		);
}