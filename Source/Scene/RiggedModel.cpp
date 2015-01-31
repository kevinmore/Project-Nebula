#include "RiggedModel.h"
#include <Scene/Scene.h>
#include <Animation/Rig/Pose.h>

RiggedModel::RiggedModel(const QString& name, Scene* scene, ModelLoaderPtr loader)
   : AbstractModel(loader->getRenderingEffect(), name),
    m_modelLoader(loader),
    m_scene(scene),
	m_skeleton(loader->getSkeletom()),
	m_FKController(0),
	m_IKSolver(0),
	m_hasAnimation(false),
	m_animationDuration(0.0f)
{
	initialize();
}

RiggedModel::RiggedModel(const QString& name, Scene* scene, ModelLoaderPtr loader, QVector<ModelDataPtr> modelData)
  : AbstractModel(loader->getRenderingEffect(), name),
    m_modelLoader(loader),
    m_scene(scene),
	m_skeleton(loader->getSkeletom()),
	m_FKController(0),
	m_IKSolver(0),
	m_hasAnimation(false),
	m_animationDuration(0.0f)
{
	m_modelDataVector = modelData;
	initialize(modelData);
}

RiggedModel::RiggedModel( const RiggedModel* orignal )
{
	m_fileName = orignal->fileName();
	m_scene = orignal->getScene();
	m_modelDataVector = orignal->getModelData();
	// copy the loader and skeleton
	m_modelLoader = orignal->getLoader();

	initialize(m_modelDataVector);

	// install shader
	QString shaderName = orignal->getShadingTech()->shaderFileName();
	m_RenderingEffect = ShadingTechniquePtr(new ShadingTechnique(m_scene, shaderName, ShadingTechnique::RIGGED));
	// copy the vao
	m_vao = orignal->getShadingTech()->getVAO();
	m_RenderingEffect->setVAO(m_vao);
}


RiggedModel::~RiggedModel() 
{
//  	SAFE_DELETE(m_skeleton);
//  	SAFE_DELETE(m_IKSolver);
// 	SAFE_DELETE(m_FKController);

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

void RiggedModel::initialize(QVector<ModelDataPtr> modelDataVector)
{
	AbstractModel::init();

	m_meshManager     = m_scene->meshManager();
	m_textureManager  = m_scene->textureManager();
	m_materialManager = m_scene->materialManager();

	// create a FKController for the model
	FKController* controller = new FKController(m_modelLoader, m_modelLoader->getSkeletom());
	setFKController(controller);
	setRootTranslation(controller->getRootTranslation());
	setRootRotation(controller->getRootRotation());

	// create an IKSolver for the model
	CCDIKSolver* solver = new CCDIKSolver(128);
	setIKSolver(solver);
	

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

	}


// 	ikSolved = false;
// 	lastUpdatedTime = 0.0f;
// 	updateIKRate = 0.5f;
// 
// 	m_FABRSolver = new FABRIKSolver(m_skeleton, 0.1f);
// 	m_FABRSolver->enableIKChain("Bip01_L_UpperArm", "Bip01_L_Hand");
// 
// 
// 	// set bone DOFs
// 	Bone* pBone;
// 	Bone::AngleLimits pitchConstraint, yawConstraint, rollConstraint;
// 	Bone::DimensionOfFreedom dof;
// 	Math::EulerAngle ea;
// 	
// 
// 	pBone = m_skeleton->getBone("Bip01_L_Clavicle");
// 	ea = pBone->getEulerAnglesInModelSpace();
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch, ea.m_fPitch);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw - 20.0f, ea.m_fYaw + 20.0f);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll - 20.0f, ea.m_fRoll + 20.0f);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 
// 	pBone = m_skeleton->getBone("Bip01_L_UpperArm");
// 	ea = pBone->getEulerAnglesInModelSpace();
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch - 50.0f, ea.m_fPitch + 50.0f);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw - 120.0f, ea.m_fYaw + 30.0f);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll - 170.0f, ea.m_fRoll + 40.0f);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 	
// 	pBone = m_skeleton->getBone("Bip01_L_Forearm");
// 	ea = pBone->getEulerAnglesInModelSpace();
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch - 5.0f, ea.m_fPitch + 120.0f);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw - 60.0f, ea.m_fYaw + 20.0f);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll, ea.m_fRoll);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 
// 	pBone = m_skeleton->getBone("Bip01_L_Hand");
// 	ea = pBone->getEulerAnglesInModelSpace();
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch - 10.0f, ea.m_fPitch + 10.0f);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw - 120.0f, ea.m_fYaw + 30.0f);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll - 5.0f, ea.m_fRoll + 5.0f);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 
// 	pBone = m_skeleton->getBone("Bip01_L_Finger2");
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch, ea.m_fPitch);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw, ea.m_fYaw);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll - 5.0f, ea.m_fRoll + 90.0f);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 
// 	pBone = m_skeleton->getBone("Bip01_L_Finger21");
// 	ea = pBone->getEulerAnglesInModelSpace();
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch, ea.m_fPitch);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw, ea.m_fYaw);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll - 5.0f, ea.m_fRoll + 90.0f);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 
// 	solvingDuration= 1.0f;
}


void RiggedModel::render( const float currentTime )
{
	float dt = currentTime - lastUpdatedTime;
	lastUpdatedTime = currentTime;
	if(dt < 0)
	{
		dt = 0.0f;
	}
	//m_actor->translateInWorld(m_actor->globalSpeed() * dt); // this is for inplace locamotion
	
	m_RenderingEffect->enable();

	QMatrix4x4 modelMatrix = m_actor->getTranformMatrix();
	modelMatrix.rotate(90, Math::Vector3::UNIT_X); // this is for dae files
	//QMatrix3x3 normalMatrix = modelViewMatrix.normalMatrix();

	m_RenderingEffect->setEyeWorldPos(m_scene->getCamera()->position());
	m_RenderingEffect->setMVPMatrix(m_scene->getCamera()->viewProjectionMatrix() * modelMatrix);
	m_RenderingEffect->setModelMatrix(modelMatrix); 
	m_RenderingEffect->setViewMatrix(m_scene->getCamera()->viewMatrix());

	// do the skeleton animation here
	// check if the model has animation first
	QVector<QMatrix4x4> Transforms;
	m_FKController->getBoneTransforms(currentTime, Transforms);

	/*
	// use IK
	// CCD
	// set constraint
	CCDIKSolver::IkConstraint constraint;
	constraint.m_startBone = m_skeleton->getBone("Bip01_L_Clavicle");
	constraint.m_endBone = m_skeleton->getBone("Bip01_L_Finger22");
	constraint.m_targetMS = m_targetPos;
	
	if (m_CCDSolver->solveOneConstraint( constraint, m_skeleton ))
	{
		m_FKController->disableBoneChain(m_skeleton->getBone("Bip01_L_Clavicle"));
		m_FKController->getBoneTransforms(time, Transforms);
		m_CCDSolver->getBoneTransforms(m_skeleton, constraint.m_startBone, Transforms);
	}
	else
	{
		m_FKController->enableAllBones();
		m_FKController->getBoneTransforms(time, Transforms);
	}
	*/

 	// update the bone positions
	for (int i = 0 ; i < Transforms.size() ; ++i) 
	{
		m_RenderingEffect->setBoneTransform(i, Transforms[i]);
	}


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

		drawElements(i);
	}

}

void RiggedModel::setReachableTargetPos( vec3& pos )
{
	m_targetPos = pos;
}

void RiggedModel::setFKController( FKController* fkCtrl )
{
	m_FKController = fkCtrl;
}

void RiggedModel::setIKSolver( CCDIKSolver* ikSolver )
{
	m_IKSolver = ikSolver;
}
