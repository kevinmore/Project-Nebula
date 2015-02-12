#include "StaticModel.h"
#include <Scene/Scene.h>

StaticModel::StaticModel(const QString& name, Scene* scene, ShadingTechniquePtr tech)
  : IModel(tech, name),
	m_scene(scene)
{
	initialize();
}

StaticModel::StaticModel(const QString& name, Scene* scene, ShadingTechniquePtr tech, QVector<ModelDataPtr> modelData)
  : IModel(tech, name),
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
	m_renderingEffect = ShadingTechniquePtr(new ShadingTechnique(shaderName, ShadingTechnique::STATIC, m_scene));
	// copy the vao
	m_vao = orignal->getShadingTech()->getVAO();
	m_renderingEffect->setVAO(m_vao);

	// copy the bounding box
	BoxColliderPtr otherBox = orignal->getBoundingBox();
	vec3 halfExtents = otherBox->getGeometryShape().getHalfExtents();

	// reset the scale of the bounding box
	vec3 scale = orignal->getScale();
	halfExtents.setX(halfExtents.x() / scale.x());
	halfExtents.setY(halfExtents.y() / scale.y());
	halfExtents.setZ(halfExtents.z() / scale.z());
	m_boundingBox  = BoxColliderPtr(new BoxCollider(otherBox->getCenter(), halfExtents, m_scene));
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
	IModel::init();

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

		// deal with the material
		// do not share the material
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
	m_renderingEffect->enable();

	m_transformMatrix = m_actor->getTransformMatrix();
	
	//QMatrix3x3 normalMatrix = modelViewMatrix.normalMatrix();
	m_renderingEffect->setEyeWorldPos(m_scene->getCamera()->position());
	m_renderingEffect->setMVPMatrix(m_scene->getCamera()->viewProjectionMatrix() * m_transformMatrix);
	m_renderingEffect->setModelMatrix(m_transformMatrix); 
	m_renderingEffect->setViewMatrix(m_scene->getCamera()->viewMatrix());

	// draw each mesh
	for(int i = 0; i < m_meshes.size(); ++i)
	{
		// bind the material
		m_materials[i]->bind();

		// enable the material
		m_renderingEffect->setMaterial(m_materials[i].data());

		drawElements(i);
	}
}
