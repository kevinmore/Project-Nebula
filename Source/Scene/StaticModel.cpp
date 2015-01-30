#include "StaticModel.h"
#include <Scene/Scene.h>

StaticModel::StaticModel(const QString& name, Scene* scene, ShadingTechniquePtr tech)
  : AbstractModel(tech, name),
	m_scene(scene)
{
	initialize();
}

StaticModel::StaticModel(const QString& name, Scene* scene, ShadingTechniquePtr tech, QVector<ModelDataPtr> modelData)
  : AbstractModel(tech, name),
    m_scene(scene)
{
	m_modelDataVector = modelData;
	initialize(modelData);
}

StaticModel::StaticModel( const StaticModel* orignal )
{
	m_fileName = orignal->fileName();
	m_scene = orignal->getScene();
	m_modelDataVector = orignal->getModelData();
	initialize(m_modelDataVector);

	// install shader
	QString shaderName = orignal->getShadingTech()->shaderFileName();
	m_RenderingEffect = ShadingTechniquePtr(new ShadingTechnique(m_scene, shaderName, ShadingTechnique::STATIC));
	// copy the vao
	m_vao = orignal->getShadingTech()->getVAO();
}


StaticModel::~StaticModel() 
{
	// clean up the meshes
	foreach(MeshPtr mesh, m_meshes)
	{
		m_meshManager->deleteMesh(mesh);
		mesh.clear();
	}

	// clean up the materials
	foreach(MaterialPtr mat, m_materials)
	{
		m_materialManager->deleteMaterial(mat);
		mat.clear();
	}
}


void StaticModel::initialize(QVector<ModelDataPtr> modelDataVector)
{
	Q_ASSERT(initializeOpenGLFunctions());


	m_meshManager     = m_scene->meshManager();
	m_textureManager  = m_scene->textureManager();
	m_materialManager = m_scene->materialManager();

	// traverse modelData vector
	for (int i = 0; i < modelDataVector.size(); ++i)
	{
		ModelDataPtr data = modelDataVector[i];

		// deal with the mesh
		MeshPtr mesh = MeshPtr(new Mesh(data->meshData.name, data->meshData.numIndices, data->meshData.baseVertex, data->meshData.baseIndex));
		m_meshes.push_back(mesh);

		// deal with the material
		MaterialPtr material(new Material(
			data->materialData.name,
			data->materialData.ambientColor,
			data->materialData.diffuseColor,
			data->materialData.specularColor,
			data->materialData.emissiveColor,
			data->materialData.shininess,
			data->materialData.shininessStrength,
			data->materialData.twoSided,
			data->materialData.blendMode,
			data->materialData.alphaBlending));

		m_materials.push_back(material);

		// deal with the texture
		TextureData td = data->materialData.textureData;
		if (!td.diffuseMap.isEmpty())
		{
			TexturePtr  texture_diffuseMap = m_textureManager->getTexture(td.diffuseMap);
			if(!texture_diffuseMap)
			{
				texture_diffuseMap = m_textureManager->addTexture(td.diffuseMap, td.diffuseMap);
			}
			material->addTexture(texture_diffuseMap);
		}
		if (!td.normalMap.isEmpty())
		{
			TexturePtr  texture_normalMap = m_textureManager->getTexture(td.normalMap);
			if(!texture_normalMap)
			{
				texture_normalMap = m_textureManager->addTexture(td.normalMap, td.normalMap, Texture::Texture2D, Texture::NormalMap);
			}
			material->addTexture(texture_normalMap);
		}
		if (!td.opacityMap.isEmpty())
		{
			TexturePtr  texture_opacityMap = m_textureManager->getTexture(td.opacityMap);
			if(!texture_opacityMap)
			{
				texture_opacityMap = m_textureManager->addTexture(td.opacityMap, td.opacityMap, Texture::Texture2D, Texture::OpacityMap);
			}
			material->addTexture(texture_opacityMap);
		}

	}

}


void StaticModel::render( float time )
{
	m_RenderingEffect->enable();

	QMatrix4x4 modelMatrix = m_actor->getTranformMatrix();
	
	//QMatrix3x3 normalMatrix = modelViewMatrix.normalMatrix();
	m_RenderingEffect->setEyeWorldPos(m_scene->getCamera()->position());
	m_RenderingEffect->setMVPMatrix(m_scene->getCamera()->viewProjectionMatrix() * modelMatrix);
	m_RenderingEffect->setModelMatrix(modelMatrix); 
	m_RenderingEffect->setViewMatrix(m_scene->getCamera()->viewMatrix());

	// draw each mesh
	for(int i = 0; i < m_meshes.size(); ++i)
	{
		// bind the material
		if (m_materials[i])
		{
			foreach(TexturePtr tex, m_materials[i]->m_textures)
			{
				if (tex->usage() == Texture::DiffuseMap)
					tex->bind(DIFFUSE_TEXTURE_UNIT);

				else if (tex->usage() == Texture::NormalMap)
					tex->bind(NORMAL_TEXTURE_UNIT);

// 				else if (tex->usage() == Texture::OpacityMap || m_materials[i]->isTranslucent())
// 				{
// 					glDepthMask(GL_FALSE);
// 					glEnable(GL_BLEND);
// 
// 					tex->bind(DIFFUSE_TEXTURE_UNIT);
// 
// 					glDisable(GL_BLEND);
// 					glDepthMask(GL_TRUE);
// 				}
			}
		}

		// enable the material
		m_RenderingEffect->setMaterial(m_materials[i]);

		drawElements(i, BaseVertex);
	}
}

void StaticModel::drawElements(unsigned int index, int mode)
{
	// Mode has not been implemented yet
	Q_UNUSED(mode);
	glBindVertexArray(m_vao);

	glDrawElementsBaseVertex(
		GL_TRIANGLES,
		m_meshes[index]->getNumIndices(),
		GL_UNSIGNED_INT,
		reinterpret_cast<void*>((sizeof(unsigned int)) * m_meshes[index]->getBaseIndex()),
		m_meshes[index]->getBaseVertex()
		);

	// Make sure the VAO is not changed from the outside    
	glBindVertexArray(0);
}