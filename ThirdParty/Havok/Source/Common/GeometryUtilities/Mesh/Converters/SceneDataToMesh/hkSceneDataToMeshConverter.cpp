/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Converters/SceneDataToMesh/hkSceneDataToMeshConverter.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/SceneData/Scene/hkxScene.h>
#include <Common/SceneData/Scene/hkxSceneUtils.h>
#include <Common/GeometryUtilities/Mesh/hkMeshTexture.h>
#include <Common/GeometryUtilities/Mesh/hkMeshMaterialRegistry.h>
#include <Common/GeometryUtilities/Mesh/Utils/VertexBufferUtil/hkMeshVertexBufferUtil.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionBuilder/hkMeshSectionBuilder.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionLockSet/hkMeshSectionLockSet.h>
#include <Common/Base/Math/Vector/hkIntVector.h>
#include <Common/Base/Math/hkMath.h>
#include <Common/Base/Math/Matrix/hkMatrix4Util.h>





extern const hkClass hkxTextureInplaceClass;
extern const hkClass hkxTextureFileClass;

template <typename TYPE>
HK_FORCE_INLINE TYPE* HK_CALL hkPtrByteOffset( void* base, int offset )
{
	return reinterpret_cast<TYPE*>( static_cast<char*>(base) + offset );
}

/* static */hkVertexFormat::ComponentUsage hkSceneDataToMeshConverter::convertUsage(hkxVertexDescription::DataUsage usage)
{
    switch (usage)
    {
        case hkxVertexDescription::HKX_DU_NONE:             return hkVertexFormat::USAGE_NONE;
        case hkxVertexDescription::HKX_DU_POSITION:         return hkVertexFormat::USAGE_POSITION;
        case hkxVertexDescription::HKX_DU_COLOR:            return hkVertexFormat::USAGE_COLOR;
        case hkxVertexDescription::HKX_DU_NORMAL:           return hkVertexFormat::USAGE_NORMAL;
        case hkxVertexDescription::HKX_DU_TANGENT:          return hkVertexFormat::USAGE_TANGENT;
        case hkxVertexDescription::HKX_DU_BINORMAL:         return hkVertexFormat::USAGE_BINORMAL;
        case hkxVertexDescription::HKX_DU_TEXCOORD:         return hkVertexFormat::USAGE_TEX_COORD;
        case hkxVertexDescription::HKX_DU_BLENDWEIGHTS:     return hkVertexFormat::USAGE_BLEND_WEIGHTS;
        case hkxVertexDescription::HKX_DU_BLENDINDICES:     return hkVertexFormat::USAGE_BLEND_MATRIX_INDEX;
        case hkxVertexDescription::HKX_DU_USERDATA:         return hkVertexFormat::USAGE_USER;
        default:                                            return hkVertexFormat::USAGE_NONE;
    }
}

/* static */void hkSceneDataToMeshConverter::convertToElement(const hkxVertexDescription::ElementDecl* decl, hkVertexFormat::Element& ele)
{
    ele.m_flags = 0;
    ele.m_subUsage = 0;

    switch (decl->m_type)
    {
		case hkxVertexDescription::HKX_DT_UINT8:
		{
			ele.m_dataType = hkVertexFormat::TYPE_UINT8;
			ele.m_numValues = decl->m_numElements;
			break;
		}
        case hkxVertexDescription::HKX_DT_INT16:
        {
            ele.m_dataType = hkVertexFormat::TYPE_INT16;
            ele.m_numValues = decl->m_numElements;
            break;
        }
        case hkxVertexDescription::HKX_DT_UINT32:
        {
            ele.m_dataType = hkVertexFormat::TYPE_UINT32;
            ele.m_numValues = decl->m_numElements;
            break;
        }
        case hkxVertexDescription::HKX_DT_FLOAT:
        {
            ele.m_dataType = hkVertexFormat::TYPE_FLOAT32;
            ele.m_numValues = decl->m_numElements;
            break;
        }
       	default:
        {
			HK_ASSERT3(0x242346f, false, "Unknown vertex format type: " << decl->m_type );
        }
    }

    ele.m_usage = convertUsage(decl->m_usage);
}

static hkMeshVertexBuffer::LockedVertices::Buffer hkSceneDataToMeshConverter_convertToBuffer(hkxVertexBuffer* vertexBuffer, const hkxVertexDescription::ElementDecl* decl)
{
    hkMeshVertexBuffer::LockedVertices::Buffer buffer;
    buffer.m_start = vertexBuffer->getVertexDataPtr( *decl );
    buffer.m_stride = decl->m_byteStride;

	hkSceneDataToMeshConverter::convertToElement(decl, buffer.m_element);
    return buffer;
}

static void hkSceneDataToMeshConverter_calculatePositions(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBufferIn, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBufferIn, const hkMatrixDecomposition::Decomposition& decomposition, int numVertices)
{
    HK_ASSERT(0x32423432, srcBufferIn.m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32 && ( (srcBufferIn.m_element.m_numValues == 3) || (srcBufferIn.m_element.m_numValues == 4) ));
	HK_ASSERT(0x32423432, dstBufferIn.m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32 && dstBufferIn.m_element.m_numValues == 3);

	hkTransform transform;

	transform.getRotation().set(decomposition.m_rotation);
	transform.getTranslation() = decomposition.m_translation;

    if ( decomposition.m_hasScale || decomposition.m_hasSkew )
    {
		hkTransform skewAndScale;

		decomposition.m_scaleAndSkew.get(skewAndScale);
		skewAndScale.getTranslation().setZero();
		hkTransform copyTransform = transform;
		transform.setMul(copyTransform, skewAndScale);
	}

	hkMeshVertexBuffer::LockedVertices::Buffer dstBuffer = dstBufferIn;
	hkMeshVertexBuffer::LockedVertices::Buffer srcBuffer = srcBufferIn;

    for (int i = 0; i < numVertices; i++)
    {
        hkVector4 pos; pos.load<3,HK_IO_NATIVE_ALIGNED>((hkFloat32*)srcBuffer.m_start);
		pos._setTransformedPos(transform, pos);

        pos.store<3,HK_IO_NATIVE_ALIGNED>((hkFloat32*)dstBuffer.m_start);

        dstBuffer.next();
        srcBuffer.next();
    }
}

static void hkSceneDataToMeshConverter_calculateNormals(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBufferIn, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBufferIn, const hkMatrixDecomposition::Decomposition& decomposition, int numVertices)
{
	HK_ASSERT(0x827baabb, srcBufferIn.m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32 && ( (srcBufferIn.m_element.m_numValues == 3) || (srcBufferIn.m_element.m_numValues == 4) ));
    HK_ASSERT(0x827baabb, dstBufferIn.m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32 && dstBufferIn.m_element.m_numValues == 3);

	hkTransform transform;

	transform.getRotation().set(decomposition.m_rotation);
	transform.getTranslation().setZero();

	if( decomposition.m_hasScale || decomposition.m_hasSkew )
    {
		hkMatrix4 scaleSkewInverseTranspose;
		hkMatrix4 sinv; hkMatrix4Util::setInverse( decomposition.m_scaleAndSkew, sinv, hkSimdReal_Eps );
		scaleSkewInverseTranspose.setTranspose(sinv);
		scaleSkewInverseTranspose.resetFourthRow();

		hkTransform skewAndScale;
		scaleSkewInverseTranspose.get(skewAndScale);
		skewAndScale.getTranslation().setZero();

		hkTransform copyTransform = transform;
		transform.setMul(copyTransform, skewAndScale);
	}

	hkMeshVertexBuffer::LockedVertices::Buffer dstBuffer = dstBufferIn;
    hkMeshVertexBuffer::LockedVertices::Buffer srcBuffer = srcBufferIn;

	for (int i = 0; i < numVertices; i++)
    {
        
        hkVector4 norm; norm.load<3,HK_IO_NATIVE_ALIGNED>((hkFloat32*)srcBuffer.m_start);
		norm._setTransformedPos(transform, norm); // we made sure above that translation is zero
        norm.normalizeIfNotZero<3>();

        norm.store<3,HK_IO_NATIVE_ALIGNED>((hkFloat32*)dstBuffer.m_start);

        dstBuffer.next();
        srcBuffer.next();
    }
}

static void hkSceneDataToMeshConverter_calculateTexCoords(const hkMeshVertexBuffer::LockedVertices::Buffer& srcBufferIn, const hkMeshVertexBuffer::LockedVertices::Buffer& dstBufferIn, int numVertices)
{
    HK_ASSERT(0x827caabb, dstBufferIn.m_element.m_dataType == hkVertexFormat::TYPE_FLOAT32 && dstBufferIn.m_element.m_numValues == 2);

    if (srcBufferIn.m_element.m_dataType == hkVertexFormat::TYPE_INT16)
    {
        hkMeshVertexBuffer::LockedVertices::Buffer dstBuffer = dstBufferIn;
        hkMeshVertexBuffer::LockedVertices::Buffer srcBuffer = srcBufferIn;

        for (int i = 0; i < numVertices; i++)
        {
            const hkInt16* srcTexCoords = (const hkInt16*)srcBuffer.m_start;
            hkFloat32* dstTexCoords = (hkFloat32*)dstBuffer.m_start;

			hkIntVector iv; iv.set(srcTexCoords[0],srcTexCoords[1],0,0);
			hkVector4 fv; iv.convertS32ToF32(fv);
			fv.mul(hkSimdReal::fromFloat(1.0f / 3276.7f));
			fv.store<2,HK_IO_NATIVE_ALIGNED>(dstTexCoords);

            dstBuffer.next();
            srcBuffer.next();
        }
    }
    else
    {
        hkMeshVertexBufferUtil::convert(srcBufferIn, dstBufferIn, numVertices);
    }
}

static void hkSceneDataToMeshConverter_setAllowMipmaps( hkMeshShape* genericMeshShape, bool allowMipmap )
{
	hkMeshSectionLockSet sectionSet;
	sectionSet.addMeshSections(genericMeshShape, 0);
	const int numSections = sectionSet.getNumSections();
	for ( int sectionIter = 0; sectionIter < numSections; sectionIter++ )
	{
		const hkMeshSection& srcSection = sectionSet.getSection(sectionIter);

		hkMeshMaterial* srcMaterial = srcSection.m_material;
		if ( srcMaterial )
		{
			for ( int textureIter = 0; textureIter < srcMaterial->getNumTextures(); textureIter++ )
			{
				hkMeshTexture* texture = srcMaterial->getTexture(textureIter);
				if ( texture != HK_NULL && !texture->isReadOnly() && texture->getHasMipMaps() != allowMipmap )
				{
					texture->setHasMipMaps(allowMipmap);
				}
			}
		}
	}
}


/* static */ hkMeshVertexBuffer* hkSceneDataToMeshConverter::convertVertexBuffer(hkMeshSystem* meshSystem, hkMatrixDecomposition::Decomposition& decomposition, hkxVertexBuffer* srcVertexBuffer )
{

	HK_ASSERT(0x7a4653a9, srcVertexBuffer );
	const hkxVertexDescription& srcVertexDesc = srcVertexBuffer->getVertexDesc();
	
    hkVertexFormat dstVertexFormat;

    const hkxVertexDescription::ElementDecl* posDecl = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_POSITION, 0);
    HK_ASSERT(0x5453fd26, posDecl);
    dstVertexFormat.addElement(hkVertexFormat::USAGE_POSITION, hkVertexFormat::TYPE_FLOAT32, 3);

    const hkxVertexDescription::ElementDecl* normDecl = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_NORMAL, 0);
    const hkxVertexDescription::ElementDecl* tangentDecl = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_TANGENT, 0);
    const hkxVertexDescription::ElementDecl* binormalDecl = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_BINORMAL, 0);
    
	const int maxVColors = 2;
	const hkxVertexDescription::ElementDecl* colorDecls[maxVColors];
	colorDecls[0] = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_COLOR, 0);
	colorDecls[1] = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_COLOR, 1);
	const hkxVertexDescription::ElementDecl* blendweightsDecl = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_BLENDWEIGHTS, 0);
    const hkxVertexDescription::ElementDecl* blendindicesDecl = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_BLENDINDICES, 0);
    
	const int maxTextureStages = 4;
    const hkxVertexDescription::ElementDecl* textureDecl[maxTextureStages];
    textureDecl[0] = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_TEXCOORD, 0);
    textureDecl[1] = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_TEXCOORD, 1);
    textureDecl[2] = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_TEXCOORD, 2);
    textureDecl[3] = srcVertexDesc.getElementDecl(hkxVertexDescription::HKX_DU_TEXCOORD, 3);

    if (normDecl)
    {
        dstVertexFormat.addElement(hkVertexFormat::USAGE_NORMAL, hkVertexFormat::TYPE_FLOAT32, 3);
    }
    if (colorDecls[0])
    {
        dstVertexFormat.addElement(hkVertexFormat::USAGE_COLOR, hkVertexFormat::TYPE_ARGB32, 1);
    }
	if (colorDecls[1])
	{
		dstVertexFormat.addElement(hkVertexFormat::USAGE_COLOR, hkVertexFormat::TYPE_ARGB32, 1);
	}
    if (tangentDecl)
    {
        dstVertexFormat.addElement(hkVertexFormat::USAGE_TANGENT, hkVertexFormat::TYPE_FLOAT32, 3);
    }
    if (binormalDecl)
    {
        dstVertexFormat.addElement(hkVertexFormat::USAGE_BINORMAL, hkVertexFormat::TYPE_FLOAT32, 3);
    }


	bool isSkin = (blendweightsDecl && blendindicesDecl);
    if (isSkin)
    {
		HK_ASSERT(0x60f996fe, blendindicesDecl->m_type == hkxVertexDescription::HKX_DT_UINT8 || blendindicesDecl->m_type == hkxVertexDescription::HKX_DT_INT16);

        // Not sure - but will assume that both are 4
        dstVertexFormat.addElement(hkVertexFormat::USAGE_BLEND_WEIGHTS, hkVertexFormat::TYPE_FLOAT32, 4);
		if(blendindicesDecl->m_type == hkxVertexDescription::HKX_DT_UINT8)
		{
			dstVertexFormat.addElement(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, hkVertexFormat::TYPE_UINT8_DWORD, 4);
		}
		else if(blendindicesDecl->m_type == hkxVertexDescription::HKX_DT_INT16)
		{
			dstVertexFormat.addElement(hkVertexFormat::USAGE_BLEND_MATRIX_INDEX, hkVertexFormat::TYPE_UINT16, 4);
		}
    }
    for (int i = 0; i < maxTextureStages; i++)
    {
        if (textureDecl[i] == HK_NULL)
        {
            break;
        }
        dstVertexFormat.addElement(hkVertexFormat::USAGE_TEX_COORD, hkVertexFormat::TYPE_FLOAT32, 2);
    }
    dstVertexFormat.makeCanonicalOrder();

    int numVertices = srcVertexBuffer->getNumVertices();

    hkMeshVertexBuffer* dstVertexBuffer = meshSystem->createVertexBuffer(dstVertexFormat, numVertices);

    if (!dstVertexBuffer)
    {
        return HK_NULL;
    }

    hkMeshVertexBuffer::LockInput lockInput;
    hkMeshVertexBuffer::LockedVertices lockedVertices;
    lockInput.m_lockFlags = hkMeshVertexBuffer::ACCESS_WRITE | hkMeshVertexBuffer::ACCESS_WRITE_DISCARD;

    hkMeshVertexBuffer::LockResult lockRes = dstVertexBuffer->lock(lockInput, lockedVertices);
    if (lockRes != hkMeshVertexBuffer::RESULT_SUCCESS)
    {
        return HK_NULL;
    }

    const int numBuffers = lockedVertices.m_numBuffers;
	for (int i = 0; i < numBuffers; i++)
    {
    
		hkMeshVertexBuffer::LockedVertices::Buffer& dstBuffer = lockedVertices.m_buffers[i];
        switch (dstBuffer.m_element.m_usage)
        {
            case hkVertexFormat::USAGE_POSITION:
            {
         		hkSceneDataToMeshConverter_calculatePositions(hkSceneDataToMeshConverter_convertToBuffer(srcVertexBuffer, posDecl), dstBuffer, decomposition, numVertices);
				break;
            }
            case hkVertexFormat::USAGE_NORMAL:
            {
                hkSceneDataToMeshConverter_calculateNormals(hkSceneDataToMeshConverter_convertToBuffer(srcVertexBuffer, normDecl), dstBuffer, decomposition, numVertices);
                break;
            }
            case hkVertexFormat::USAGE_TANGENT:
            {
                hkSceneDataToMeshConverter_calculateNormals(hkSceneDataToMeshConverter_convertToBuffer(srcVertexBuffer, tangentDecl), dstBuffer, decomposition, numVertices);
                break;
            }
            case hkVertexFormat::USAGE_BINORMAL:
            {
                hkSceneDataToMeshConverter_calculateNormals(hkSceneDataToMeshConverter_convertToBuffer(srcVertexBuffer, binormalDecl), dstBuffer, decomposition, numVertices);
                break;
            }
            case hkVertexFormat::USAGE_TEX_COORD:
            {
                const hkxVertexDescription::ElementDecl* texDecl = textureDecl[dstBuffer.m_element.m_subUsage];
				
				// NOTE! Think hkx has bug in that its returning that a tex coord is one tex coord

				hkMeshVertexBuffer::LockedVertices::Buffer srcBuffer = hkSceneDataToMeshConverter_convertToBuffer(srcVertexBuffer, texDecl);
				srcBuffer.m_element.m_numValues = 2;
                hkSceneDataToMeshConverter_calculateTexCoords(srcBuffer, dstBuffer,  numVertices);
                break;
            }
            case hkVertexFormat::USAGE_BLEND_MATRIX_INDEX:
            {
                hkMeshVertexBufferUtil::convert(hkSceneDataToMeshConverter_convertToBuffer(srcVertexBuffer, blendindicesDecl), dstBuffer,  numVertices);
                break;
            }
            case hkVertexFormat::USAGE_BLEND_WEIGHTS:
           {
                hkMeshVertexBufferUtil::convert(hkSceneDataToMeshConverter_convertToBuffer(srcVertexBuffer, blendweightsDecl), dstBuffer,  numVertices);
                break;
            }
            case hkVertexFormat::USAGE_COLOR:
            {
				const hkxVertexDescription::ElementDecl* colorDecl = colorDecls[dstBuffer.m_element.m_subUsage];
				hkMeshVertexBufferUtil::convert(hkSceneDataToMeshConverter_convertToBuffer(srcVertexBuffer, colorDecl), dstBuffer,  numVertices);
				break;
            }
            default: break;
        }

    }

    dstVertexBuffer->unlock(lockedVertices);

    return dstVertexBuffer;

}

/* static */ hkResult HK_CALL hkSceneDataToMeshConverter::convertIndices(const hkxMeshSection& section, hkMeshSectionBuilder& builder, hkMeshVertexBuffer* vertexBuffer, hkMeshMaterial* dstMaterial)
{
    const int numIndexBuffers = section.m_indexBuffers.getSize();      

    for (int i = 0; i < numIndexBuffers; i++)
    {
        const hkxIndexBuffer& indexBuffer = *section.m_indexBuffers[i];				

        hkMeshSection::PrimitiveType primType = hkMeshSection::PRIMITIVE_TYPE_UNKNOWN;
        switch (indexBuffer.m_indexType)
        {
            case hkxIndexBuffer::INDEX_TYPE_TRI_LIST:
            {
                primType = hkMeshSection:: PRIMITIVE_TYPE_TRIANGLE_LIST;
                break;
            }
            case hkxIndexBuffer::INDEX_TYPE_TRI_STRIP:
            {
                primType = hkMeshSection:: PRIMITIVE_TYPE_TRIANGLE_STRIP;
                break;
            }
            default: break;
        }

        if (primType == hkMeshSection::PRIMITIVE_TYPE_UNKNOWN)
        {
            HK_ASSERT(0x34324, !"Unsupported index type");
            continue;
        }

		builder.startMeshSection(vertexBuffer, dstMaterial);

		if ((indexBuffer.m_indices16.getSize() == 0) && (indexBuffer.m_indices32.getSize() == 0))
		{
			// This has no indices - so turn into indices
			builder.concatUnindexed(primType, indexBuffer.m_vertexBaseOffset, indexBuffer.m_length);
		}
		else
		{
			if (indexBuffer.m_indices32.getSize()  > 0)
			{
				builder.concatPrimitives(primType, indexBuffer.m_indices32.begin(), indexBuffer.m_indices32.getSize());
			}
			else if(indexBuffer.m_indices16.getSize() > 0)
			{
				builder.concatPrimitives(primType, indexBuffer.m_indices16.begin(), indexBuffer.m_indices16.getSize());
			}
			else
			{
				HK_WARN_ALWAYS(0x4d18dcee, "Couldn't handle indices");
			}
		}
		
		builder.endMeshSection();
    }

	return HK_SUCCESS;
}


/* static */ hkMeshShape* hkSceneDataToMeshConverter::convert(hkMeshSystem* meshSystem, hkMeshMaterial* overrideMaterial, const class hkxScene* scene, const hkxNode* node, Space space, bool allowMipmaps)
{
    hkxMesh* mesh = hkxSceneUtils::getMeshFromNode( node );
    if (!mesh)
    {
        return HK_NULL;
    }
    hkMatrix4 worldTransform;
	switch (space)
	{
	    case SPACE_LOCAL:
		    worldTransform.setIdentity();
		    break;
	    case SPACE_TO_WORLD_SPACE:
		    scene->getWorldFromNodeTransform(node, worldTransform);
		    break;
	    case SPACE_ONLY_USING_SCALE_SKEW:
		{
			if (HK_SUCCESS == scene->getWorldFromNodeTransform(node, worldTransform))
			{
				hkMatrixDecomposition::Decomposition decomposition;
				hkMatrixDecomposition::decomposeMatrix( worldTransform, decomposition );

				if (decomposition.m_hasScale || decomposition.m_hasSkew)
				{
					const hkVector4& c0 = decomposition.m_scaleAndSkew.getColumn<0>();
					const hkVector4& c1 = decomposition.m_scaleAndSkew.getColumn<1>();
					const hkVector4& c2 = decomposition.m_scaleAndSkew.getColumn<2>();
					worldTransform.setCols( c0,c1,c2,hkVector4::getZero());
				}
				else
				{
					worldTransform.setIdentity();
				}
			}
			else
			{
				HK_WARN_ALWAYS(0xabba8572, "cannot retrieve node transform");
				worldTransform.setIdentity();
			}
			break;
		}
	}
	hkMeshShape* shape = convert(meshSystem, overrideMaterial, worldTransform, mesh, allowMipmaps);
	if (shape)
	{
		shape->setName( node->m_name );
	}
	return shape;
}

/* static */ hkMeshShape* hkSceneDataToMeshConverter::convert(hkMeshSystem* meshSystem, hkMeshMaterial* overrideMaterial, const hkMatrix4& worldTransform, hkxMesh* mesh, bool allowMipmaps)
{
    int numSections = mesh->m_sections.getSize();
	if (numSections < 1)
    {
        return HK_NULL;
    }

	hkRefPtr<hkxMeshSection>* sections = mesh->m_sections.begin();

	// See if it has scale
	// As Havok rigid bodies do not have scale, when we use
	// meshes with rigid bodies the scale must be either baked in
	// or set as an extra transform in the display object.
	// For now I will just bake it in as instancing is not the norm
	hkMatrixDecomposition::Decomposition decomposition;
	hkMatrixDecomposition::decomposeMatrix(worldTransform, decomposition);

	// We don't do any merging by material here - it can be performed with the output using the hkMeshSectionMergeUtil

	hkPointerMap<hkxVertexBuffer*, hkMeshVertexBuffer*> vertexBufferMap;
	hkInplaceArray<hkMeshVertexBuffer*,16> vertexBuffers;

	hkMeshSectionBuilder builder;
	
	// For each section
	for (int i = 0; i < numSections; ++i)
	{
		const hkxMeshSection& section = *sections[i];
		hkxMaterial* srcMaterial = section.m_material;

		// We need to have a material - to use internally
		hkMeshMaterial* dstMaterial = HK_NULL;
		if (overrideMaterial)
		{
			dstMaterial = overrideMaterial;
		}
		else
		{
			if (srcMaterial && srcMaterial->m_name)
			{
				dstMaterial = meshSystem->getMaterialRegistry()->findMaterial(srcMaterial->m_name);
			}

			if(dstMaterial == HK_NULL && srcMaterial != HK_NULL)
			{								
				dstMaterial = convert(meshSystem, srcMaterial);
				
				if( dstMaterial != HK_NULL )
				{
					meshSystem->getMaterialRegistry()->registerMaterial(dstMaterial->getName(), dstMaterial);
					dstMaterial->removeReference();
				}				
			}
		}

		hkMeshVertexBuffer* vertexBuffer = vertexBufferMap.getWithDefault(section.m_vertexBuffer, HK_NULL);
		if (!vertexBuffer)
		{
			// create a vertex buffer from our hkxVertexBuffer
			vertexBuffer = convertVertexBuffer(meshSystem, decomposition, section.m_vertexBuffer);

			if (!vertexBuffer)
			{
				HK_WARN_ALWAYS(0x4729259, "Mesh section does not produce a vertex buffer\n");
				hkReferencedObject::removeReferences(vertexBuffers.begin(), vertexBuffers.getSize());
				return HK_NULL;
			}
			vertexBufferMap.insert(section.m_vertexBuffer, vertexBuffer);
			vertexBuffers.pushBack(vertexBuffer);
		}
		
		convertIndices(section, builder, vertexBuffer, dstMaterial);
	}

	// Create the result
	hkMeshShape* meshShape = meshSystem->createShape(builder.getSections(), builder.getNumSections());

	// Enforce mipmap setting
	hkSceneDataToMeshConverter_setAllowMipmaps( meshShape, allowMipmaps );

	// No longer need the vertex buffer references
	hkReferencedObject::removeReferences(vertexBuffers.begin(), vertexBuffers.getSize());

	// Done
	return meshShape;
}

/* static */ hkMeshMaterial* hkSceneDataToMeshConverter::convert(hkMeshSystem* meshSystem, hkxMaterial* material, bool allowMipmaps)
{
	hkMeshMaterial* result = meshSystem->createMaterial();	
	
	hkStringBuf materialName;	
	materialName.append(material->m_name);

	for(int i = 0; i < material->m_stages.getSize(); ++i)
	{
		hkMeshTexture* destinationTexture = meshSystem->createTexture();
		{
			destinationTexture->setHasMipMaps(allowMipmaps);
			destinationTexture->setFilterMode(hkMeshTexture::ANISOTROPIC);
			destinationTexture->setUsageHint((hkMeshTexture::TextureUsageType)((hkInt32)material->m_stages[i].m_usageHint));
			destinationTexture->setTextureCoordChannel(material->m_stages[i].m_tcoordChannel);
		}
		
		if( hkxTextureInplaceClass.equals(material->m_stages[i].m_texture.getClass()) )
		{
			hkxTextureInplace* sourceTexture = static_cast<hkxTextureInplace*>(material->m_stages[i].m_texture.val());
			
			hkMeshTexture::Format format;

			if(hkString::strNcasecmp("PNG", sourceTexture->m_fileType, 3) == 0)
			{
				format = hkMeshTexture::PNG;
			}
			else if(hkString::strNcasecmp("TGA", sourceTexture->m_fileType, 3) == 0)
			{
				format = hkMeshTexture::TGA;
			}
			else if(hkString::strNcasecmp("BMP", sourceTexture->m_fileType, 3) == 0)
			{
				format = hkMeshTexture::BMP;
			}
			else if(hkString::strNcasecmp("DDS", sourceTexture->m_fileType, 3) == 0)
			{
				format = hkMeshTexture::DDS;
			}
			else
			{
				format = hkMeshTexture::Unknown;
			}

			if( format != hkMeshTexture::Unknown )
			{
				destinationTexture->setData(sourceTexture->m_data.begin(), sourceTexture->m_data.getSize(), format);
			}
			else
			{
				HK_WARN(0x344d591a, "Unsupported texture format found, not converting texture." );
				result->removeReference();
				destinationTexture->removeReference();
				return HK_NULL;
			}						
		}
		else if( hkxTextureFileClass.equals(material->m_stages[i].m_texture.getClass()) )
		{
			hkxTextureFile* sourceTexture = static_cast<hkxTextureFile*>(material->m_stages[i].m_texture.val());
			destinationTexture->setFilename(sourceTexture->m_filename);
		}
		else
		{
			HK_WARN(0x2f177390, "Unsupported texture class.  Only inplace textures are supported." );
			result->removeReference();
			destinationTexture->removeReference();
			return HK_NULL;
		}

		result->addTexture(destinationTexture);
		destinationTexture->removeReference();
	}
	
	result->setColors(material->m_diffuseColor, material->m_ambientColor, material->m_specularColor, material->m_emissiveColor);
	result->setName(materialName.cString());
	result->setUserData(material->m_userData);

	// Search for additional displacement mapping parameters
	hkReferencedObject* displAmount = material->findAttributeObjectByName("DisplacementAmount");
	if ( displAmount && hkxAnimatedFloatClass.equals(displAmount->getClassType()) )
	{
		hkxAnimatedFloat* flt = reinterpret_cast<hkxAnimatedFloat*>(displAmount);
		if ( flt->m_floats.getSize() )
		{
			result->setDisplacementAmount(flt->m_floats[0]);
		}
	}

	// Search for additional tesselation parameters
	hkReferencedObject* tessFactor = material->findAttributeObjectByName("TesselationFactor");
	if ( tessFactor && hkxAnimatedFloatClass.equals(tessFactor->getClassType()) )
	{
		hkxAnimatedFloat* flt = reinterpret_cast<hkxAnimatedFloat*>(tessFactor);
		if ( flt->m_floats.getSize() )
		{
			result->setTesselationFactor(flt->m_floats[0]);
		}
	}

	return result;
}

//	Retrieves the vertex positions from a hkMeshShape

/*static*/ void HK_CALL hkSceneDataToMeshConverter::collectVertexPositions(const hkMeshShape* mesh, hkArray<hkVector4>& verticesInOut)
{
	int numSections = mesh->getNumSections();
	for (int si = 0 ; si < numSections; si++)
	{
		// Get mesh section si
		hkMeshSection crtMeshSection;

		mesh->lockSection(si, hkMeshShape::ACCESS_VERTEX_BUFFER, crtMeshSection);
		{
			hkMeshVertexBuffer * vertices = crtMeshSection.m_vertexBuffer;

			// Get vertex buffer format
			hkVertexFormat vtxFmt;
			vertices->getVertexFormat(vtxFmt);

			// Get position index inside the vertex buffer
			int positionIdx = vtxFmt.findElementIndex(hkVertexFormat::USAGE_POSITION, 0);

			// Lock vertex buffer
			hkMeshVertexBuffer::LockInput		lockIn;
			hkMeshVertexBuffer::LockedVertices	lockedVerts;
			vertices->lock(lockIn, lockedVerts);
			{
				int numVerts = vertices->getNumVertices();

				// Retrieve the raw positions from the vertex buffer
				hkVector4 * destPtr = verticesInOut.expandBy(numVerts);
				hkArray<hkFloat32>::Temp va; va.setSize(4*numVerts);
				vertices->getElementVectorArray(lockedVerts, positionIdx, va.begin());
				for (int i=0; i<numVerts; ++i)
				{
					destPtr[i].load<4,HK_IO_NATIVE_ALIGNED>(&va[4*i]);
				}
			}
			vertices->unlock(lockedVerts);
		}
		mesh->unlockSection(crtMeshSection);
	}
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
