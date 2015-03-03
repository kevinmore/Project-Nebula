/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Converters/MeshToSceneData/hkMeshToSceneDataConverter.h>
#include <Common/GeometryUtilities/Mesh/hkMeshBody.h>
#include <Common/GeometryUtilities/Mesh/hkMeshTexture.h>
#include <Common/SceneData/Scene/hkxScene.h>
#include <Common/GeometryUtilities/Mesh/Skin/hkSkinnedRefMeshShape.h>
#include <Common/Base/Reflection/hkClass.h>

template <typename TYPE>
HK_FORCE_INLINE TYPE* HK_CALL hkPtrByteOffset( void* base, int offset )
{
	return reinterpret_cast<TYPE*>( static_cast<char*>(base) + offset );
}

/* static */hkxVertexDescription::DataUsage hkMeshToSceneDataConverter::convertUsage(hkVertexFormat::ComponentUsage usage)
{
    switch (usage)
    {
		case hkVertexFormat::USAGE_NONE: return hkxVertexDescription::HKX_DU_NONE;
		case hkVertexFormat::USAGE_POSITION: return hkxVertexDescription::HKX_DU_POSITION;
		case hkVertexFormat::USAGE_COLOR: return hkxVertexDescription::HKX_DU_COLOR;
		case hkVertexFormat::USAGE_NORMAL: return hkxVertexDescription::HKX_DU_NORMAL;
        case hkVertexFormat::USAGE_TANGENT: return hkxVertexDescription::HKX_DU_TANGENT;
        case hkVertexFormat::USAGE_BINORMAL: return hkxVertexDescription::HKX_DU_BINORMAL;
        case hkVertexFormat::USAGE_TEX_COORD: return hkxVertexDescription::HKX_DU_TEXCOORD;
        case hkVertexFormat::USAGE_BLEND_WEIGHTS: return hkxVertexDescription::HKX_DU_BLENDWEIGHTS;
		case hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED: return hkxVertexDescription::HKX_DU_BLENDWEIGHTS;
        case hkVertexFormat::USAGE_BLEND_MATRIX_INDEX: return hkxVertexDescription::HKX_DU_BLENDINDICES;
        case hkVertexFormat::USAGE_USER: return hkxVertexDescription::HKX_DU_USERDATA;
		default: return hkxVertexDescription::HKX_DU_NONE;
    }
}

/* static */void hkMeshToSceneDataConverter::convertToDecl( const hkVertexFormat::Element& elem, hkxVertexDescription::ElementDecl& decl)
{
	decl.m_type = hkxVertexDescription::HKX_DT_NONE;

	switch (elem.m_dataType)
    {
		case hkVertexFormat::TYPE_UINT8:
		{
			decl.m_type = hkxVertexDescription::HKX_DT_UINT8;
			decl.m_numElements = elem.m_numValues;
			break;
		}

		case hkVertexFormat::TYPE_INT16: 
        {
            decl.m_type = hkxVertexDescription::HKX_DT_INT16;
			decl.m_numElements = elem.m_numValues;
            break;
        }
		case hkVertexFormat::TYPE_ARGB32:	
		case hkVertexFormat::TYPE_UINT32:
        {  
            decl.m_type = hkxVertexDescription::HKX_DT_UINT32;
			decl.m_numElements = elem.m_numValues;
            break;
        }
		case hkVertexFormat::TYPE_FLOAT32: 
        {
			decl.m_type = hkxVertexDescription::HKX_DT_FLOAT;
			decl.m_numElements = elem.m_numValues;
            break;
        }
		case hkVertexFormat::TYPE_UINT8_DWORD:
			{
				decl.m_type = hkxVertexDescription::HKX_DT_UINT8;
				decl.m_numElements = elem.m_numValues;
			}
			break;
        default:
        {
            HK_WARN_ALWAYS(0x24234, "Unknown vertex format data type. Don't know what to do with it - discarding.");
        }
    }

    decl.m_usage = convertUsage( elem.m_usage );
}

static inline int _elemSize( const hkxVertexDescription::ElementDecl& declOut )
{
	switch (declOut.m_usage)
	{
		case hkxVertexDescription::HKX_DU_POSITION:
		case hkxVertexDescription::HKX_DU_NORMAL:
		case hkxVertexDescription::HKX_DU_TANGENT:
		case hkxVertexDescription::HKX_DU_BINORMAL:
			return 3*sizeof(hkFloat32);
		
		case hkxVertexDescription::HKX_DU_TEXCOORD:
			if (declOut.m_type == hkxVertexDescription::HKX_DT_FLOAT)
				return declOut.m_numElements*sizeof(hkFloat32);
			else if (declOut.m_type == hkxVertexDescription::HKX_DT_INT16)
				return declOut.m_numElements*sizeof(hkInt16);
			break;

		case hkxVertexDescription::HKX_DU_COLOR:
			if (declOut.m_type == hkxVertexDescription::HKX_DT_FLOAT)
				return declOut.m_numElements*sizeof(hkFloat32);
			else if (declOut.m_type == hkxVertexDescription::HKX_DT_UINT32)
				return declOut.m_numElements*sizeof(hkUint32);
			break;
		case hkxVertexDescription::HKX_DU_BLENDINDICES:
			if (declOut.m_type == hkxVertexDescription::HKX_DT_UINT8)
				return declOut.m_numElements*sizeof(hkUint8);
			if (declOut.m_type == hkxVertexDescription::HKX_DT_INT16)
				return declOut.m_numElements*sizeof(hkUint16);
		case hkxVertexDescription::HKX_DU_BLENDWEIGHTS:
			if (declOut.m_type == hkxVertexDescription::HKX_DT_UINT8)
				return declOut.m_numElements*sizeof(hkUint8);
			if (declOut.m_type == hkxVertexDescription::HKX_DT_INT16)
				return declOut.m_numElements*sizeof(hkUint16);
			if (declOut.m_type == hkxVertexDescription::HKX_DT_FLOAT)
				return declOut.m_numElements*sizeof(hkFloat32);
		case hkxVertexDescription::HKX_DU_USERDATA:
		case hkxVertexDescription::HKX_DU_NONE:
		default:
			break;
	}
	return 0;
}

static inline void _copyVert( hkUint8* currentVertexOut, const hkUint8* currentVertexIn, const hkVertexFormat::Element& elemIn, const hkxVertexDescription::ElementDecl& declOut )
{
	if ((elemIn.m_dataType == hkVertexFormat::TYPE_NONE) || (declOut.m_type == hkxVertexDescription::HKX_DT_NONE))
	{
		//then unknown or no match, so ignore
		return;
	}

	hkString::memCpy( currentVertexOut, currentVertexIn, _elemSize(declOut));
}

static void _reinterpretVert( hkUint8* currentVertexOut, const hkUint8* currentVertexIn, const hkVertexFormat::Element& elemIn, const hkxVertexDescription::ElementDecl& declOut )
{
	//XX Currently don't support reinterprets (will place breakpoints and see what is needed (if any)
	hkUint8* data = currentVertexOut + declOut.m_byteOffset;
		
	hkString::memSet(data, 0, _elemSize(declOut));
	
	bool hasReinterpreted = false;

	switch ( elemIn.m_dataType )
	{
	case hkVertexFormat::TYPE_FLOAT32:
		if ( (elemIn.m_numValues < 4) && (declOut.m_type == hkxVertexDescription::HKX_DT_FLOAT) )
		{
			hkString::memCpy( data, currentVertexIn, elemIn.m_numValues * sizeof(hkFloat32) );
			hasReinterpreted = true;
		}
		break;

	case hkVertexFormat::TYPE_VECTOR4:
		if ( (elemIn.m_numValues == 1) && (declOut.m_type == hkxVertexDescription::HKX_DT_FLOAT) )
		{
			hkString::memCpy( data, currentVertexIn, 4 * sizeof(hkFloat32) );
			hasReinterpreted = true;
		}
		break;

	case hkVertexFormat::TYPE_UINT8_DWORD:
		{
			// Most likely bone indices
			HK_ASSERT(0xf602232, elemIn.m_numValues == 4);
			if ( declOut.m_type == hkxVertexDescription::HKX_DT_UINT8 )
			{
				HK_ASSERT(0x29add6f9, declOut.m_numElements == 4);
				data[0] = currentVertexIn[0];
				data[1] = currentVertexIn[1];
				data[2] = currentVertexIn[2];
				data[3] = currentVertexIn[3];
				hasReinterpreted = true;
			}
		}
		break;

	default:
		break;
	}

	if ( !hasReinterpreted )
	{
		HK_WARN_ONCE(0xdd88f26, "Failed to _reinterpretVert! Rendering will be broken!!!");
	}
}

static inline bool _usageMatch(const hkxVertexDescription::DataUsage& hkxUsage, const hkVertexFormat::ComponentUsage& hkUsage)
{
	switch (hkUsage)
	{
		case hkVertexFormat::USAGE_POSITION: return hkxUsage == hkxVertexDescription::HKX_DU_POSITION; 
		case hkVertexFormat::USAGE_NORMAL: return hkxUsage == hkxVertexDescription::HKX_DU_NORMAL; 
		case hkVertexFormat::USAGE_COLOR: return hkxUsage == hkxVertexDescription::HKX_DU_COLOR; 
		case hkVertexFormat::USAGE_TANGENT: return hkxUsage == hkxVertexDescription::HKX_DU_TANGENT; 
		case hkVertexFormat::USAGE_BINORMAL: return hkxUsage == hkxVertexDescription::HKX_DU_BINORMAL; 
		case hkVertexFormat::USAGE_BLEND_MATRIX_INDEX: return hkxUsage == hkxVertexDescription::HKX_DU_BLENDINDICES; 
		case hkVertexFormat::USAGE_BLEND_WEIGHTS: // fall through 
		case hkVertexFormat::USAGE_BLEND_WEIGHTS_LAST_IMPLIED: return hkxUsage == hkxVertexDescription::HKX_DU_BLENDWEIGHTS; 
		case hkVertexFormat::USAGE_TEX_COORD: return hkxUsage == hkxVertexDescription::HKX_DU_TEXCOORD; 
		case hkVertexFormat::USAGE_USER: return hkxUsage == hkxVertexDescription::HKX_DU_USERDATA; 
		
		case hkVertexFormat::USAGE_POINT_SIZE: // fall through 
		default: return false; 
	}
}

static inline bool _dataFormatMatch( const hkVertexFormat::Element& elem, const hkxVertexDescription::ElementDecl& decl )
{
	switch (elem.m_dataType)
	{
	case hkVertexFormat::TYPE_UINT8: return (decl.m_type == hkxVertexDescription::HKX_DT_UINT8);	
	
	case hkVertexFormat::TYPE_INT16: return (decl.m_type == hkxVertexDescription::HKX_DT_INT16);	
	
	case hkVertexFormat::TYPE_UINT32: return (decl.m_type == hkxVertexDescription::HKX_DT_UINT32);	
	
	case hkVertexFormat::TYPE_UINT8_DWORD:  return (decl.m_type == hkxVertexDescription::HKX_DT_UINT32);	                 
	
	case hkVertexFormat::TYPE_ARGB32: return (decl.m_type == hkxVertexDescription::HKX_DT_UINT32);						
	
	case hkVertexFormat::TYPE_FLOAT32:
#if !defined(HK_REAL_IS_DOUBLE)
	case hkVertexFormat::TYPE_VECTOR4: 
#endif
		return (decl.m_type == hkxVertexDescription::HKX_DT_FLOAT);
		
	case hkVertexFormat::TYPE_INT8:
	case hkVertexFormat::TYPE_UINT16: 
	case hkVertexFormat::TYPE_INT32: 
	case hkVertexFormat::TYPE_FLOAT16: 
	default:
		return false;
	}
}

static void _findRemapping( const hkMeshVertexBuffer::LockedVertices& lo, const hkArray<hkxVertexDescription::ElementDecl>& decls, hkArray<int>& remap, hkArray<bool>& dataFormatMatch)
{
	//for each decl, find the matching buffer elem and store the buffer index 
	for (int di=0; di < decls.getSize(); ++di)
	{
		const hkxVertexDescription::ElementDecl& dc = decls[di];
		int bi = 0;
		for (; bi < lo.m_numBuffers; ++bi)
		{
			const hkMeshVertexBuffer::LockedVertices::Buffer& b = lo.m_buffers[bi];
			if ( _usageMatch(dc.m_usage, b.m_element.m_usage) && (remap.indexOf(bi) < 0) )
			{
				dataFormatMatch.pushBack( _dataFormatMatch( b.m_element, dc ) );
				remap.pushBack(bi);
				break;
			}
		}

		if ( bi == lo.m_numBuffers) // no match
		{
			dataFormatMatch.pushBack(false);
			remap.pushBack(-1);
		}
	}
}

/*static*/ hkxVertexBuffer* HK_CALL hkMeshToSceneDataConverter::convertVertexBuffer( hkMeshVertexBuffer* buf )
{
	hkMeshVertexBuffer::LockInput li;
	li.m_startVertex = 0;
	li.m_numVertices = -1;
	li.m_lockFlags = hkMeshVertexBuffer::ACCESS_READ;
	hkMeshVertexBuffer::LockedVertices lvo; 

	if (hkMeshVertexBuffer::RESULT_SUCCESS == buf->lock( li, lvo ))
	{
		if (lvo.m_numBuffers < 1)
		{
			return HK_NULL;
		}

		// see if we can match the vb close enough
		hkxVertexBuffer* newVB = new hkxVertexBuffer; 
		hkxVertexDescription vdescDesired;
		
		hkUlong basePtr = ~hkUlong(0x0); 
		for (int bi=0; bi < lvo.m_numBuffers; ++bi)
		{
			hkMeshVertexBuffer::LockedVertices::Buffer& b = lvo.m_buffers[bi];
			convertToDecl( b.m_element, vdescDesired.m_decls.expandOne() );
			if ((hkUlong)b.m_start < basePtr)
			{
				basePtr = (hkUlong)b.m_start;
			}
		}
		
		newVB->setNumVertices( lvo.m_numVertices, vdescDesired );
		const hkxVertexDescription& vdescActual = newVB->getVertexDesc();


		hkArray<int> bufferMap;
		hkArray<bool> dataMatch;

		_findRemapping( lvo, vdescActual.m_decls, bufferMap, dataMatch);
		
		for (int vi=0; vi < lvo.m_numVertices; ++vi)
		{
			for (int ve=0; ve < vdescActual.m_decls.getSize(); ++ve)
			{
				int bi = bufferMap[ve];
				if (bi < 0)
				{
					continue;
				}
				
				const hkxVertexDescription::ElementDecl& eDesc = vdescActual.m_decls[ve];
				hkUint8* currentVertexOut = static_cast<hkUint8*>( newVB->getVertexDataPtr(eDesc) ) + (vi * eDesc.m_byteStride);
				hkMeshVertexBuffer::LockedVertices::Buffer& lockedBuf = lvo.m_buffers[bi];
				
				if ( dataMatch[ve] )
				{
					_copyVert( currentVertexOut, (hkUint8*)lockedBuf.m_start, lockedBuf.m_element, eDesc );
				}
				else
				{
					_reinterpretVert( currentVertexOut, (hkUint8*)lockedBuf.m_start, lockedBuf.m_element, eDesc );
				}

				lockedBuf.next();
			}
		}

		buf->unlock(lvo);
		return newVB;
	}

	return HK_NULL;
}

/*static*/ hkxIndexBuffer* HK_CALL hkMeshToSceneDataConverter::convertIndexBuffer(const hkMeshSection* section )
{
	if (!section || (section->m_numIndices == 0))
	{
		return HK_NULL;
	}

	hkxIndexBuffer* newBuffer = new hkxIndexBuffer;
	if (section->m_indexType == hkMeshSection::INDEX_TYPE_UINT16)
	{
		newBuffer->m_indices16.setSize( section->m_numIndices );
		hkString::memCpy(newBuffer->m_indices16.begin(), section->m_indices, section->m_numIndices * hkSizeOf(hkUint16));
	}
	else 
	{
		newBuffer->m_indices32.setSize( section->m_numIndices );
		hkString::memCpy(newBuffer->m_indices32.begin(), section->m_indices, section->m_numIndices * hkSizeOf(hkUint32));
	}

	newBuffer->m_vertexBaseOffset = section->m_indices ? 0 : section->m_vertexStartIndex; 
	newBuffer->m_length = section->m_numIndices;

	switch (section->m_primitiveType)
	{
		case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_LIST:
			newBuffer->m_indexType = hkxIndexBuffer::INDEX_TYPE_TRI_LIST;
			break;

		case hkMeshSection::PRIMITIVE_TYPE_TRIANGLE_STRIP:
			newBuffer->m_indexType = hkxIndexBuffer::INDEX_TYPE_TRI_STRIP;
			break;

		default:
			newBuffer->m_indexType = hkxIndexBuffer::INDEX_TYPE_INVALID;
			break;
	}

	return newBuffer;
}


static hkxMaterial* _recursivelySearchForMaterial(const char* materialName, hkxMaterial* xMaterial)
{
	if ( xMaterial->m_name == materialName )
	{
		return xMaterial;
	}

	{
		for (int smi=0; smi < xMaterial->m_subMaterials.getSize(); smi++)
		{
			hkxMaterial* matchingSubMaterial = _recursivelySearchForMaterial(materialName, xMaterial->m_subMaterials[smi]);
			if ( matchingSubMaterial )
			{
				return matchingSubMaterial;
			}
		}
	}

	return HK_NULL;
}

/*static*/ hkxMesh* HK_CALL hkMeshToSceneDataConverter::convertShape(hkxScene* scene, const hkMeshShape* shape)
{
	// Get scene materials
	const hkArray< hkRefPtr<hkxMaterial> >& sceneMtls = scene->m_materials;
	const int numSceneMtls = sceneMtls.getSize();

	// In the case of a skinref, convert the mesh buffer
	
	if ( hkSkinnedRefMeshShapeClass.equals(shape->getClassType()) )
	{
		hkSkinnedMeshShape* skinnedShape = static_cast<const hkSkinnedRefMeshShape*>(shape)->getSkinnedMeshShape();
		hkSkinnedMeshShape::BoneSection boneSection;
		skinnedShape->getBoneSection( 0, boneSection );
		shape = boneSection.m_meshBuffer;
	}

	hkArray< hkRefPtr<hkxMeshSection> > newSections;
	for ( int si =0; si < shape->getNumSections(); ++si ) 
	{
		hkMeshSection section;
		shape->lockSection( si, hkMeshShape::ACCESS_INDICES | hkMeshShape::ACCESS_VERTEX_BUFFER, section );
		if (section.m_vertexBuffer)
		{

			hkxVertexBuffer* newBuf = convertVertexBuffer( section.m_vertexBuffer );
			if (newBuf )
			{
				hkxIndexBuffer* newIBuf = convertIndexBuffer( &section );
				if (newIBuf)
				{
					hkxMeshSection* newSect = new hkxMeshSection();
					
					newSect->m_vertexBuffer = newBuf;
					newBuf->removeReference();
					
					newSect->m_indexBuffers.setSize(1);
					newSect->m_indexBuffers[0] = newIBuf;
					newIBuf->removeReference();
					
					newSect->m_material = HK_NULL;
					if ( section.m_material )
					{
						// Try to locate the material among the scene materials
						const char* materialName = section.m_material->getName();
						for (int m = 0; m < numSceneMtls; m++)
						{
							hkxMaterial* mat = _recursivelySearchForMaterial(materialName, sceneMtls[m]);
							if ( mat )
							{
								newSect->m_material = mat;
								break;
							}
						}

						// If we found nothing, convert now, must be a cloned material
						if ( !newSect->m_material )
						{
							// Create scene material
							hkxMaterial* newMtl = hkMeshToSceneDataConverter::convertMaterial(scene, section.m_material);
							newSect->m_material = newMtl;
							scene->m_materials.pushBack(newMtl);
							newMtl->removeReference();
						}
					}

					newSections.pushBack(newSect);
					newSect->removeReference();
				}
				else
				{
					newBuf->removeReference();
				}
			}
		}

		shape->unlockSection(section);
	}

	if (newSections.getSize() > 0)
	{
		hkxMesh* newMesh = new hkxMesh;
		newMesh->m_sections.append(newSections.begin(), newSections.getSize());
		return newMesh;
	}

	return HK_NULL;
}

/*static*/ hkxNode* HK_CALL hkMeshToSceneDataConverter::convertBody(hkxScene* scene, const hkMeshBody* body)
{
	if (!body || !body->getMeshShape())
	{
		return HK_NULL;
	}

	hkxMesh* newMesh = convertShape(scene, body->getMeshShape());
	if (newMesh)
	{
		hkxNode* newNode = new hkxNode;
		newNode->m_object = newMesh;

		newNode->m_name = body->getName();
		
		newNode->m_keyFrames.setSize(1);
		body->getTransform( newNode->m_keyFrames[0] );
		
		newNode->m_selected = false;

		return newNode;
	}

	return HK_NULL;
}

/*static */ hkxMaterial* HK_CALL hkMeshToSceneDataConverter::convertMaterial(hkxScene* scene, const hkMeshMaterial* material)
{
	if ( !material )
	{
		return HK_NULL;
	}

	// Get scene textures
	hkArray< hkRefPtr<hkxTextureFile> >& externalTextures	= scene->m_externalTextures;
	hkArray< hkRefPtr<hkxTextureInplace> >& inplaceTextures	= scene->m_inplaceTextures;

	// Alloc result
	hkxMaterial* result = new hkxMaterial();

	// Resolve material textures
	for (int i = 0; i < material->getNumTextures(); i++)
	{
		hkMeshTexture* srcTexture	= material->getTexture(i);
		const char* srcTextureName	= srcTexture->getFilename();
		hkxMaterial::TextureStage& desTexture = result->m_stages.expandOne();
		
		const int texChannel		= srcTexture->getTextureCoordChannel();
		desTexture.m_tcoordChannel	= (texChannel < 0) ? i : texChannel;
		desTexture.m_usageHint		= (hkxMaterial::TextureType)srcTexture->getUsageHint(); //XX CK: not safe really, hkxMaterial could easily expand in future (blend masks etc)
		desTexture.m_texture		= HK_NULL;

		// Try to match texture against the scene's external textures
		for (int ti = externalTextures.getSize() - 1; ti >= 0; ti--)
		{
			hkxTextureFile* tex = externalTextures[ti];
			if ( tex->m_filename.compareTo(srcTextureName) == 0 )
			{
				desTexture.m_texture = tex;
				break;
			}
		}
		if ( desTexture.m_texture )
		{
			continue;
		}

		// Try to match texture against the scene's embedded textures
		for (int ti = inplaceTextures.getSize() - 1; ti >= 0; ti--)
		{
			hkxTextureInplace* tex = inplaceTextures[ti];
			if ( tex->m_name.compareTo(srcTextureName) == 0 )
			{
				desTexture.m_texture = tex;
				break;
			}
		}
		if ( desTexture.m_texture )
		{
			continue;
		}

		// Nothing found so far. If this is a RAW texture, embed it now
		{
			hkUint8* data;
			int size;
			hkMeshTexture::Format format;
			srcTexture->getData(data, size, format);

			if ( format == hkMeshTexture::RAW )
			{
				hkxTextureInplace* tex = new hkxTextureInplace();
				
				tex->m_data.append(data, size);
				tex->m_name.set(srcTextureName);
				tex->m_originalFilename.set("");
				
				hkString::strCpy(tex->m_fileType, "RAW");
				desTexture.m_texture = tex;
				scene->m_inplaceTextures.pushBack(tex);
				tex->removeReference();
			}
		}
	}

	// Copy all other material parameters
	material->getColors(result->m_diffuseColor, result->m_ambientColor, result->m_specularColor, result->m_emissiveColor);
	result->m_name		= material->getName();
	result->m_userData	= material->getUserData();

	// If we have a non-zero displacement amount, save it on the material!
	const hkReal displacementAmount = material->getDisplacementAmount();
	if ( displacementAmount && !result->findAttributeObjectByName("DisplacementAmount") )
	{
		hkxAnimatedFloat* flt = new hkxAnimatedFloat();
		flt->m_floats.pushBack((hkFloat32)displacementAmount);

		hkxAttribute attrib;
		attrib.m_name		= "DisplacementAmount";
		attrib.m_value		= flt;

		hkxAttributeGroup attribGroup;
		attribGroup.m_name	= "Displacement";
		attribGroup.m_attributes.pushBack(attrib);

		result->m_attributeGroups.pushBack(attribGroup);
		flt->removeReference();
	}

	// If we have a non-zero tesselation factor, save it on the material!
	const hkReal tesselationFactor = material->getTesselationFactor();
	if ( tesselationFactor && !result->findAttributeObjectByName("TesselationFactor") )
	{
		hkxAnimatedFloat* flt = new hkxAnimatedFloat();
		flt->m_floats.pushBack((hkFloat32)tesselationFactor);

		hkxAttribute attrib;
		attrib.m_name		= "TesselationFactor";
		attrib.m_value		= flt;

		hkxAttributeGroup attribGroup;
		attribGroup.m_name	= "Tesselation";
		attribGroup.m_attributes.pushBack(attrib);

		result->m_attributeGroups.pushBack(attribGroup);
		flt->removeReference();
	}

	return result;
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
