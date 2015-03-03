/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Scene/hkxSceneUtils.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/SceneData/Environment/hkxEnvironment.h>
#include <Common/SceneData/Mesh/Channels/hkxVertexFloatDataChannel.h>
#include <Common/SceneData/Scene/hkxScene.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

// perform the given transform on the selected scene elements
void hkxSceneUtils::transformScene( hkxScene& scene, const SceneTransformOptions& opts )
{
	// Construct the "transformInfo" object
	TransformInfo transformInfo;
	{
		// The 4x4 matrix
		transformInfo.m_transform = opts.m_transformMatrix;

		// Its inverse
		transformInfo.m_inverse = transformInfo.m_transform;
		if( transformInfo.m_inverse.invert( HK_REAL_EPSILON ) != HK_SUCCESS )
		{
			HK_WARN_ALWAYS ( 0xabba45e4, "Inversion failed. Check the Matrix is not singular" );
			return;
		}

		// The inverse, transposed (for normals)
		transformInfo.m_inverseTranspose = transformInfo.m_inverse;
		transformInfo.m_inverseTranspose.transpose();

		// Its decomposition
		hkMatrixDecomposition::decomposeMatrix(transformInfo.m_transform, transformInfo.m_decomposition);
	}

	HK_REPORT_SECTION_BEGIN(0x5e4345e4, "Transform Scene" );

	// nodes
	if( opts.m_applyToNodes && scene.m_rootNode )
	{
		// transform node and node's children
		hkxSceneUtils::transformNode( transformInfo, *scene.m_rootNode);

		HK_REPORT( "Processed all nodes in the scene." );
	}

	// skin bindings
	if( opts.m_applyToNodes && scene.m_skinBindings.getSize() )
	{
		// iterate through all binding in the scene
		for( int cs = 0; cs < scene.m_skinBindings.getSize(); ++cs )
		{
			hkxSkinBinding* skin= scene.m_skinBindings[cs];
			hkxSceneUtils::transformSkinBinding( transformInfo, *skin);
		}

		HK_REPORT( "Processed " << scene.m_skinBindings.getSize() << " skin bindings." );
	}

	// mesh buffers (vertex and/or index)
	if( ( opts.m_applyToBuffers || opts.m_flipWinding ) && scene.m_meshes.getSize() )
	{
		// find the buffers
		for( int cm = 0; cm < scene.m_meshes.getSize(); ++cm )
		{
			hkxMesh* curMesh = scene.m_meshes[cm];

			for( int cs = 0; cs < curMesh->m_sections.getSize(); ++cs )
			{
				hkxMeshSection* curSection = curMesh->m_sections[cs];

				// buffers?
				if( opts.m_applyToBuffers )
				{
					// transform the vertex buffer
					hkxSceneUtils::transformVertexBuffer( transformInfo, *curSection->m_vertexBuffer);
				}

				// winding?
				if( opts.m_flipWinding )
				{
					for( int cib = 0; cib < curSection->m_indexBuffers.getSize(); ++cib )
					{
						// flip the triangle winding
						hkxSceneUtils::flipWinding( *curSection->m_indexBuffers[cib] );
					}
				}
			}
		}

		HK_REPORT( "Processed " << scene.m_meshes.getSize() << " meshes." );
	}

	// vertex float channels 
	if ( opts.m_applyToFloatChannels )
	{
		for( int cm = 0; cm < scene.m_meshes.getSize(); ++cm )
		{
			hkxMesh* curMesh = scene.m_meshes[cm];

			// find the float channels
			for (int ci=0; ci<curMesh->m_userChannelInfos.getSize(); ++ci)
			{
				HK_ASSERT(0x610862a4, curMesh->m_userChannelInfos[ci]);
				const hkxMesh::UserChannelInfo& info = *curMesh->m_userChannelInfos[ci];

				if ( hkString::strCmp(info.m_className, hkxVertexFloatDataChannelClass.getName()) == 0 )
				{
					// transform this vertex float channel in each mesh section
					for (int si=0; si<curMesh->m_sections.getSize(); ++si)
					{
						hkxMeshSection* section = curMesh->m_sections[si];
						hkxVertexFloatDataChannel* floatChannel = static_cast<hkxVertexFloatDataChannel*>( section->m_userChannels[ci].val() );
						hkxSceneUtils::transformFloatChannel( transformInfo, *floatChannel );
					}
				}
			}
		}
	}
	else if (transformInfo.m_decomposition.m_hasScale)
	{
		// HCL-737 : Warn user if distance/angle channels found and using a scaling transform
		bool ok = true;
		for( int cm = 0; ok && (cm < scene.m_meshes.getSize()); ++cm )
		{
			hkxMesh* curMesh = scene.m_meshes[cm];

			// find the float channels
			for (int ci=0; ok && (ci<curMesh->m_userChannelInfos.getSize()); ++ci)
			{
				HK_ASSERT(0x37fd0201, curMesh->m_userChannelInfos[ci]);
				const hkxMesh::UserChannelInfo& info = *curMesh->m_userChannelInfos[ci];

				if ( hkString::strCmp(info.m_className, hkxVertexFloatDataChannelClass.getName()) == 0 )
				{
					
					if ( hkString::strCmp(info.m_className, hkxVertexFloatDataChannelClass.getName()) == 0 )
					{
						// transform this vertex float channel in each mesh section
						for (int si=0; ok && si<curMesh->m_sections.getSize(); ++si)
						{
							hkxMeshSection* section = curMesh->m_sections[si];
							hkxVertexFloatDataChannel* floatChannel = static_cast<hkxVertexFloatDataChannel*>( section->m_userChannels[ci].val() );
							if (floatChannel->m_dimensions!=hkxVertexFloatDataChannel::FLOAT)
							{
								HK_WARN_ALWAYS(0xabba8ea1, "Vertex channels found but not transformed - was this intentional?");
								ok = false;
							}
						}
					}
				}
			}
		}
	}

	// cameras
	if( opts.m_applyToCameras && scene.m_cameras.getSize() )
	{
		for( int cc = 0; cc < scene.m_cameras.getSize(); ++cc )
		{
			// transform the cameras
			hkxSceneUtils::transformCamera( transformInfo, *scene.m_cameras[cc]);
		}

		HK_REPORT( "Processed " << scene.m_cameras.getSize() << " cameras." );
	}

	// lights
	if( opts.m_applyToLights && scene.m_lights.getSize() )
	{
		for( int cc = 0; cc < scene.m_lights.getSize(); ++cc )
		{
			// transform the lights
			hkxSceneUtils::transformLight( transformInfo, *scene.m_lights[cc]);
		}

		HK_REPORT( "Processed " << scene.m_lights.getSize()  << " lights." );
	}

        // Finally, the scene itself, remembering that transforms in the pipeline in the 
	// order Ta -> Tb should be multiplied as Tb * Ta   (EXP-1438)
	
	hkMatrix3 Ta = scene.m_appliedTransform;  // the incoming transform
	hkMatrix3 Tb = transformInfo.m_transform; // the transform in this filter
	Tb.mul(Ta); 
	scene.m_appliedTransform = Tb;
	HK_REPORT_SECTION_END();
}


// transform a node's keys
void hkxSceneUtils::transformNode( const TransformInfo& transformInfo, hkxNode& node)
{
	// recurse into node tree
	for( int c = 0; c < node.m_children.getSize(); ++c )
	{
		hkxSceneUtils::transformNode( transformInfo, *node.m_children[c]);
	}

	// Transform the keyframes
	for( int i = 0; i < node.m_keyFrames.getSize(); i++ )
	{
		transformMatrix4(transformInfo, node.m_keyFrames[i]);
	}

	// Transform any attributes of the node that can be transformed
	for ( int agIndex = 0; agIndex < node.m_attributeGroups.getSize(); ++agIndex)
	{
		hkxAttributeGroup& ag = node.m_attributeGroups[agIndex];

		hkxSceneUtils::transformAttributeGroup (transformInfo, ag);
	}
}

void hkxSceneUtils::transformAttributeGroup(const TransformInfo& transformInfo, hkxAttributeGroup& attributeGroup)
{
	for ( int aIndex = 0; aIndex < attributeGroup.m_attributes.getSize(); ++aIndex)
	{
		hkxAttribute& attribute = attributeGroup.m_attributes[aIndex];

		if (attribute.m_value)
		{
			// we can transform certain known attribute types
			// and can transform based on the hint in those attributes
			const hkClass* klass = attribute.m_value.getClass();

			// Floats
			if ( hkString::strCmp( klass->getName(), hkxAnimatedFloatClass.getName()) == 0)
			{
				hkxAnimatedFloat* f = (hkxAnimatedFloat*) attribute.m_value.val();
				transformAnimatedFloat (transformInfo, *f);

				continue;
			}

			// Vectors
			if ( hkString::strCmp( klass->getName(), hkxAnimatedVectorClass.getName()) == 0)
			{
				hkxAnimatedVector* v = (hkxAnimatedVector*)attribute.m_value.val();
				transformAnimatedVector (transformInfo, *v);

				continue;
			}

			// Quaternions
			if ( hkString::strCmp( klass->getName(), hkxAnimatedQuaternionClass.getName()) == 0)
			{
				hkxAnimatedQuaternion* q = (hkxAnimatedQuaternion*)attribute.m_value.val();
				transformAnimatedQuaternion (transformInfo, *q);

				continue;
			}

			// Matrices
			if ( hkString::strCmp( klass->getName(), hkxAnimatedMatrixClass.getName()) == 0)
			{
				hkxAnimatedMatrix* m = (hkxAnimatedMatrix*)attribute.m_value.val();
				transformAnimatedMatrix (transformInfo, *m);

				continue;

			}
		}
	}
}

void hkxSceneUtils::transformAnimatedFloat (const TransformInfo& transformInfo, hkxAnimatedFloat& animatedFloat)
{
	// Scale : floats representing distances
	const bool shouldScale = (animatedFloat.m_hint & hkxAttribute::HINT_SCALE) != 0;

	// Flip : floats representing angles
	const bool shouldFlip = ((animatedFloat.m_hint & hkxAttribute::HINT_FLIP) && transformInfo.m_decomposition.m_flips) != 0;

	// Floats can only be scaled or flipped
	if ( shouldScale || shouldFlip)
	{
		if (shouldScale && shouldFlip)
		{
			HK_WARN_ALWAYS(0xabba8a03, "Float attribute with both FLIP and SCALE flags... Weird..");
		}

		hkFloat32 scaleFloat = (shouldFlip) ? -1.0f : 1.0f;
		if (shouldScale)
		{
			const hkVector4& scaleVector = transformInfo.m_decomposition.m_scale;
			const hkFloat32 scaleLength = hkFloat32(scaleVector.length<3>().getReal()) * 0.57735026918962576450914878050196f; // 1/ sqrt(3)
			scaleFloat *= scaleLength;

		}

		for (int fi= 0; fi < animatedFloat.m_floats.getSize(); ++fi)
		{
			animatedFloat.m_floats[fi] *= scaleFloat;
		}
	}
}

void hkxSceneUtils::transformAnimatedQuaternion (const TransformInfo& transformInfo,  hkxAnimatedQuaternion& animatedQuaternion)
{
	// Quaternions are always transformed as they always represent rotations
	for (int qi=0; qi < animatedQuaternion.m_quaternions.getSize()/4; qi++)
	{
		hkQuaternion quatRef; 
		quatRef.m_vec.load<4,HK_IO_NATIVE_ALIGNED>(&animatedQuaternion.m_quaternions[4*qi]);

		// We rotate the axis of the quaternion by the basis of the transform
		hkVector4 imag = quatRef.getImag();
		imag._setRotatedDir(transformInfo.m_decomposition.m_basis, imag);
		quatRef.setImag(imag);

		// And if the transformation involves a change of handedness, flip the sign 
		if (transformInfo.m_decomposition.m_flips)
		{
			quatRef.setRealPart(-quatRef.getRealPart());
		}

		quatRef.m_vec.store<4,HK_IO_NATIVE_ALIGNED>(&animatedQuaternion.m_quaternions[4*qi]);
	}
}

void hkxSceneUtils::transformAnimatedMatrix (const TransformInfo& transformInfo,  hkxAnimatedMatrix& animatedMatrix)
{
	if (animatedMatrix.m_hint & hkxAttribute::HINT_TRANSFORM_AND_SCALE)
	{
		for (int mi= 0; mi < animatedMatrix.m_matrices.getSize()/16; ++mi)
		{
			hkMatrix4 m; 
			m.set4x4ColumnMajor(&animatedMatrix.m_matrices[16*mi]);
			transformMatrix4(transformInfo, m);
			m.get4x4ColumnMajor(&animatedMatrix.m_matrices[16*mi]);
		}
	}
}

void hkxSceneUtils::transformAnimatedVector (const TransformInfo& transformInfo,  hkxAnimatedVector& animatedVector)
{
	// Vectors are either just rotated or rotated and scaled
	const bool shouldRotate = (animatedVector.m_hint & hkxAttribute::HINT_TRANSFORM) != 0;
	const bool shouldScale = (animatedVector.m_hint & hkxAttribute::HINT_SCALE) != 0;

	if (!shouldRotate && !shouldScale)
	{
		return;
	}

	hkMatrix3 theTransform; theTransform.setIdentity();
	{
		if (shouldRotate && !shouldScale)
		{
			theTransform = transformInfo.m_decomposition.m_basis;
		}

		if (shouldRotate && shouldScale)
		{
			theTransform = transformInfo.m_transform;
		}

		if (!shouldRotate && shouldScale)
		{
			// uncommon, but...
			const hkVector4& scaleVector = transformInfo.m_decomposition.m_scale;
			hkMatrix3Util::_setDiagonal(scaleVector, theTransform);
		}
	}

	for (int vi= 0; vi < animatedVector.m_vectors.getSize()/4; ++vi)
	{
		hkVector4 vs;
		vs.load<4,HK_IO_NATIVE_ALIGNED>(&animatedVector.m_vectors[4*vi]);
		vs._setRotatedDir(theTransform, vs);
		vs.store<4,HK_IO_NATIVE_ALIGNED>(&animatedVector.m_vectors[4*vi]);
	}
}



// transform the skin binding
void hkxSceneUtils::transformSkinBinding( const TransformInfo& transformInfo, hkxSkinBinding& nodeInfo)
{
	// Transform initBoneTransform Array
	{
		for( int i = 0; i < nodeInfo.m_bindPose.getSize(); i++ )
		{
			transformMatrix4(transformInfo, nodeInfo.m_bindPose[i]);
		}
	}

	// Transform init skin transform
	{
		transformMatrix4 (transformInfo, nodeInfo.m_initSkinTransform);
	}
}


// transform the vertex buffer data
void hkxSceneUtils::transformVertexBuffer( const TransformInfo& transformInfo, hkxVertexBuffer& vb)
{
	// Positions are transformed by the full matrix
	const hkMatrix3& trans = transformInfo.m_transform;

	// Normals are transformed by the inverse of the basis, transposed
	const hkMatrix3& transNormal = transformInfo.m_inverseTranspose;

	const hkxVertexDescription& desc = vb.getVertexDesc();
	const hkxVertexDescription::ElementDecl* posDecl =  desc.getElementDecl( hkxVertexDescription::HKX_DU_POSITION, 0 );
	const hkxVertexDescription::ElementDecl* normDecl =  desc.getElementDecl( hkxVertexDescription::HKX_DU_NORMAL, 0 );
	const hkxVertexDescription::ElementDecl* tangDecl =  desc.getElementDecl( hkxVertexDescription::HKX_DU_TANGENT, 0 );
	const hkxVertexDescription::ElementDecl* binormDecl = desc.getElementDecl( hkxVertexDescription::HKX_DU_BINORMAL, 0 );
	int posStride = 0;
	int normStride = 0;
	int tangStride = 0;
	int binormStride = 0;
	void* posData = HK_NULL;
	void* normData = HK_NULL;
	void* tangData = HK_NULL;
	void* binormData = HK_NULL;
	if (posDecl)
	{
		posData = vb.getVertexDataPtr(*posDecl);
		posStride = posDecl->m_byteStride;
	}
	if (normDecl)
	{
		normData = vb.getVertexDataPtr(*normDecl);
		normStride = normDecl->m_byteStride;
	}
	if (tangDecl)
	{
		tangData = vb.getVertexDataPtr(*tangDecl);
		tangStride = tangDecl->m_byteStride;
	}
	if (binormDecl)
	{
		binormData = vb.getVertexDataPtr(*binormDecl);
		binormStride = binormDecl->m_byteStride;
	}
	
	int numVerts = vb.getNumVertices();
	for( int i = 0; i < numVerts; i++ )
	{
		if (posDecl)
		{
			hkFloat32* data = (hkFloat32*)posData;
			hkVector4 vector; 
			vector.load<3,HK_IO_NATIVE_ALIGNED>(data);
			vector._setRotatedDir( trans, vector );
			vector.setComponent<3>(hkSimdReal_1);
			vector.store<4,HK_IO_NATIVE_ALIGNED>(data);
		}

		if (normDecl)
		{
			hkFloat32* data = (hkFloat32*)normData;
			hkVector4 vector; 
			vector.load<3,HK_IO_NATIVE_ALIGNED>(data);
			vector._setRotatedDir( transNormal, vector );
			vector.normalizeIfNotZero<3>();
			vector.zeroComponent<3>();
			vector.store<4,HK_IO_NATIVE_ALIGNED>(data);
		}

		if (tangDecl)
		{
			hkFloat32* data = (hkFloat32*)tangData;
			hkVector4 vector; 
			vector.load<3,HK_IO_NATIVE_ALIGNED>(data);
			vector._setRotatedDir( transNormal, vector );
			vector.normalizeIfNotZero<3>();
			vector.zeroComponent<3>();
			vector.store<4,HK_IO_NATIVE_ALIGNED>(data);
		}

		if (binormDecl)
		{
			hkFloat32* data = (hkFloat32*)binormData;
			hkVector4 vector; 
			vector.load<3,HK_IO_NATIVE_ALIGNED>(data);
			vector._setRotatedDir( transNormal, vector );
			vector.normalizeIfNotZero<3>();
			vector.zeroComponent<3>();
			vector.store<4,HK_IO_NATIVE_ALIGNED>(data);
		}
		
		posData = hkAddByteOffset(posData, posStride);
		normData = hkAddByteOffset(normData, normStride);
		tangData = hkAddByteOffset(tangData, tangStride);
		binormData = hkAddByteOffset(binormData, binormStride);
	}

}

void hkxSceneUtils::transformVertexBuffer( const hkTransform& tr, class hkxVertexBuffer& vbuffer)
{
	// Do the rotation part first
	// Now do the translation part
	int numVerts = vbuffer.getNumVertices();
	if (numVerts > 0)
	{
		hkxSceneUtils::TransformInfo transformInfo;
		{
			// The 4x4 matrix
			transformInfo.m_transform = tr.getRotation();

			// Its inverse
			transformInfo.m_inverse = transformInfo.m_transform;
			if( transformInfo.m_inverse.invert( HK_REAL_EPSILON ) != HK_SUCCESS )
			{
				HK_WARN_ALWAYS ( 0xabba45e4, "Inversion failed. Check the Matrix is not singular" );
				return;
			}

			// The inverse, transposed (for normals)
			transformInfo.m_inverseTranspose = transformInfo.m_inverse;
			transformInfo.m_inverseTranspose.transpose();

			// Its decomposition
			hkMatrixDecomposition::decomposeMatrix(transformInfo.m_transform, transformInfo.m_decomposition);
		}

		hkxSceneUtils::transformVertexBuffer(transformInfo, vbuffer);

		// then translate:
		const hkxVertexDescription& desc = vbuffer.getVertexDesc();
		const hkxVertexDescription::ElementDecl* posDecl =  desc.getElementDecl( hkxVertexDescription::HKX_DU_POSITION, 0 );
		if (posDecl)
		{
			hkFloat32* pos = (hkFloat32*)vbuffer.getVertexDataPtr(*posDecl);
			int posStride = posDecl->m_byteStride / sizeof(hkFloat32);
			for (int vi=0; vi < numVerts; ++vi)
			{
				hkVector4 p;
				p.load<3,HK_IO_NATIVE_ALIGNED>(pos);
				p.add( tr.getTranslation() );
				p.setComponent<3>(hkSimdReal_1);
				p.store<4,HK_IO_NATIVE_ALIGNED>(pos);
				pos += posStride;
			}
		}
	}
}

// transform the vertex float channel
void hkxSceneUtils::transformFloatChannel(const TransformInfo& transformInfo, hkxVertexFloatDataChannel& floatChannel)
{
	switch (floatChannel.m_dimensions)
	{
		case hkxVertexFloatDataChannel::DISTANCE:
		{
			const hkVector4& scaleVector = transformInfo.m_decomposition.m_scale;
			const hkFloat32 scaleFloat = hkFloat32(scaleVector.length<3>().getReal()) * 0.57735026918962576450914878050196f; // 1 / sqrt(3)

			for (int fi=0; fi<floatChannel.m_perVertexFloats.getSize(); ++fi)
			{
				hkFloat32 &f = floatChannel.m_perVertexFloats[fi];
				f *= scaleFloat;
			}
		}
		break;

		case hkxVertexFloatDataChannel::ANGLE:
		{
			const bool shouldFlip = transformInfo.m_decomposition.m_flips;
			if (shouldFlip)
			{
				for (int fi=0; fi<floatChannel.m_perVertexFloats.getSize(); ++fi)
				{
					float &f = floatChannel.m_perVertexFloats[fi];
					f *= -1.0f;
				}
			}
		}
		break;

		case hkxVertexFloatDataChannel::FLOAT:
		break;
	}
}


// transform the light
void hkxSceneUtils::transformLight( const TransformInfo& transformInfo, hkxLight& light)
{
	light.m_position._setRotatedDir( transformInfo.m_transform, light.m_position );
	light.m_direction._setRotatedDir( transformInfo.m_decomposition.m_basis, light.m_direction );
}


// transform the camera
void hkxSceneUtils::transformCamera( const TransformInfo& transformInfo, hkxCamera& camera)
{
	camera.m_from._setRotatedDir( transformInfo.m_transform, camera.m_from );
	camera.m_focus._setRotatedDir( transformInfo.m_transform, camera.m_focus );
	camera.m_up._setRotatedDir( transformInfo.m_decomposition.m_basis, camera.m_up );

	// Get a single float value for the scale in the transform,  
	// and scale the clipping planes by it
	const hkVector4& scaleVector = transformInfo.m_decomposition.m_scale;
	const hkReal scaleLength = scaleVector.length<3>().getReal() * hkReal(0.57735026918962576450914878050196f); // 1 / sqrt(3)
	camera.m_near *= scaleLength;
	camera.m_far *= scaleLength;

	// Change handness of the camera if required
	if (transformInfo.m_decomposition.m_flips)
	{
		camera.m_leftHanded = !camera.m_leftHanded;
	}

}


// flip the triangle winding
void hkxSceneUtils::flipWinding( hkxIndexBuffer &ibuffer )
{
	// 16 bit indices
	int numI = ibuffer.m_indices16.getSize() | ibuffer.m_indices32.getSize();
	if( ibuffer.m_indices16.getSize())
	{
		switch ( ibuffer.m_indexType )
			{
			case hkxIndexBuffer::INDEX_TYPE_TRI_LIST:
			{
				for( int i = 0; i < numI; i += 3 )
				{
					hkAlgorithm::swap( ibuffer.m_indices16[i+0], ibuffer.m_indices16[i+2] );
				}
				break;
			}
			case hkxIndexBuffer::INDEX_TYPE_TRI_STRIP:
			{
				// winding implicit off the first tri
				hkAlgorithm::swap( ibuffer.m_indices16[1], ibuffer.m_indices16[2] ); 
				break;
			}
			// fan not handled
			default: break; 
		}
	}

	// 32 bit indices
	else if (ibuffer.m_indices32.getSize())
	{
		switch ( ibuffer.m_indexType )
		{
			case hkxIndexBuffer::INDEX_TYPE_TRI_LIST:
				{
					for( int i = 0; i < numI; i += 3 )
					{
						hkAlgorithm::swap( ibuffer.m_indices32[i+0], ibuffer.m_indices32[i+2] );
					}
					break;
				}
			case hkxIndexBuffer::INDEX_TYPE_TRI_STRIP:
				{
					// winding implicit off the first tri
					hkAlgorithm::swap( ibuffer.m_indices32[1], ibuffer.m_indices32[2] ); 
					break;
				}
				// fan not handled
			default: break; 
		}
	}
	//else just tri order in the Vertex Buffer.. TODO: Not handled either.

}

// Transforms a fullMatrix4, reused in different places
void hkxSceneUtils::transformMatrix4 (const TransformInfo& transformInfo, hkMatrix4& matrix4)
{
	// We split the matrix4 into a matrix3 and a translation
	hkMatrix3 matrix3; 
	{
		// Take it from the matrix4
		matrix3.setCols(matrix4.getColumn<0>(), matrix4.getColumn<1>(), matrix4.getColumn<2>());

		// Change of basis (t * m * t^-1)
		hkMatrix3 temp;
		temp.setMul(matrix3, transformInfo.m_inverse);
		matrix3.setMul(transformInfo.m_transform, temp);
	}
	
	hkVector4 translation;
	{
		translation = matrix4.getColumn<3>();

		translation._setRotatedDir(transformInfo.m_transform, translation);
		translation.setW(hkSimdReal_1);
	}

	// We put it back together
	matrix4.setCols(matrix3.getColumn<0>(), matrix3.getColumn<1>(), matrix3.getColumn<2>(), translation);
}

static hkxNode* _findFirstSelected( hkxNode* node )
{
	if( !node )
	{
		return HK_NULL;
	}

	for( int i=0; i<node->m_children.getSize(); ++i )
	{
		hkxNode* child = node->m_children[i];
		if( child->m_selected )
		{
			return child;
		}
		else
		{
			hkxNode* descendant = _findFirstSelected( child );
			if( descendant != HK_NULL )
			{
				return descendant;
			}
		}
	}
	return HK_NULL;
}

void hkxSceneUtils::fillEnvironmentFromScene(const hkxScene& scene, hkxEnvironment& environment)
{
	if (scene.m_modeller)
	{
		environment.setVariable("modeller", scene.m_modeller);
	}

	if (scene.m_asset)
	{
		// Full path
		{
			environment.setVariable("assetPath", scene.m_asset);
		}

		// Find the asset name, i.e. remove the path and file extension.
		{
			hkStringBuf assetName( scene.m_asset );
			{
				int offset = assetName.lastIndexOf( '\\' ) + 1;
				int len = ( assetName.lastIndexOf( '.' ) - offset > 0 ) ? assetName.lastIndexOf( '.' ) - offset : assetName.getLength();

				assetName.slice( offset, len );

				if( assetName.getLength() == 0 )
				{
					assetName = "untitled";
				}
			}

			environment.setVariable("asset", assetName.cString());
		}

		// Folder
		{
			hkStringBuf fullPath(scene.m_asset);

			const int folderEndIdx = hkMath::max2(fullPath.lastIndexOf('\\'), fullPath.lastIndexOf('/'));

			hkStringBuf assetFolder = "";

			if (folderEndIdx>=0)
			{
				assetFolder = hkStringBuf(scene.m_asset, folderEndIdx+1);
			}

			environment.setVariable("assetFolder", assetFolder.cString());
		}


	}

	// EXP-631
	{
		hkxNode* firstSelected = _findFirstSelected(scene.m_rootNode);

		if (firstSelected && firstSelected->m_name)
		{
			environment.setVariable("selected", firstSelected->m_name);
		}
	}

	
}

static bool _nodeCompare (hkxNode* one, hkxNode* two)
{
	if (!one || !one->m_name) return true;
	if (!two || !two->m_name) return false;

	return (hkString::strCasecmp(one->m_name, two->m_name)<0);
}

static void _reorderChildrenOfNodeRecursive (hkxNode* node)
{
	if (!node) return;

	// Recurse
	{
		for (int i=0; i<node->m_children.getSize(); i++)
		{
			_reorderChildrenOfNodeRecursive(node->m_children[i]);
		}
	}

	// Reorder
	{
		hkAlgorithm::quickSort(node->m_children.begin(), node->m_children.getSize(), _nodeCompare);
	}
}

/*static*/ void hkxSceneUtils::reorderNodesAlphabetically ( class hkxScene& scene )
{
	_reorderChildrenOfNodeRecursive(scene.m_rootNode);
}

/*static*/ hkxMesh* hkxSceneUtils::getMeshFromNode (const hkxNode* node)
{
	if (!node) return HK_NULL;
	if (!node->m_object) return HK_NULL;

	if (hkString::strCasecmp (node->m_object.getClass()->getName(), hkxMeshClass.getName())==0)
	{
		return (hkxMesh*) node->m_object.val();
	}
	else if (hkString::strCasecmp(node->m_object.getClass()->getName(), hkxSkinBindingClass.getName())==0)
	{
		return ((hkxSkinBinding*) node->m_object.val())->m_mesh;
	}

	return HK_NULL;
}

const hkxNode* HK_CALL hkxSceneUtils::findFirstNodeUsingMesh(const hkxNode* node, const hkxMesh* aMesh)
{
	if (node)
	{
		if (node->m_object.getClass())
		{
			hkxMesh* nodeMesh = getMeshFromNode( node );
			if (nodeMesh == aMesh)
			{
				return node;
			}
		}
	
		for (int ni=0; ni < node->m_children.getSize(); ++ni)
		{
			const hkxNode* foundNode = findFirstNodeUsingMesh( node->m_children[ni], aMesh );
			if (foundNode)
			{
				return foundNode;
			}
		}
	}

	return HK_NULL;
	
}




void HK_CALL hkxSceneUtils::findAllGraphicsNodes(bool collectShapes, bool ignorePShapes, const hkStringMap<int>& extraNodesToFind, hkxNode* node, hkArray<GraphicsNode>& nodesOut)
{
	if ( node->findAttributeGroupByName("hkRigidBody") )
	{
		ignorePShapes = false;
		collectShapes = true;
	}
	if ( node->findAttributeGroupByName("hkdBody") )
	{
		ignorePShapes = false;
		collectShapes = true;
	}
	if ( node->findAttributeGroupByName("hkdShape") )
	{
		ignorePShapes = false;
		collectShapes = true;
	}

	if ( extraNodesToFind.hasKey( node->m_name ) )
	{
		ignorePShapes = false;
		collectShapes = true;
	}

	if ( node->findAttributeGroupByName("hkShape") ) 
	{
		if ( ignorePShapes )
		{
			return;
		}
		collectShapes = true;
	}

	//bool graphicsFound = false;
	if ( collectShapes && !ignorePShapes )
	{
		hkxMesh* mesh = hkxSceneUtils::getMeshFromNode( node );
		if ( mesh )
		{
			GraphicsNode& n = nodesOut.expandOne();
			n.m_node = node;
			n.m_name = node->m_name; 
			//graphicsFound = true;
		}
	}


	{
		for (int c=0; c < node->m_children.getSize(); ++c)
		{
			hkxNode* child = node->m_children[c];
			findAllGraphicsNodes( collectShapes, ignorePShapes, extraNodesToFind, child, nodesOut);
		}
	}
}

void HK_CALL hkxSceneUtils::findAllNodes(hkxNode* node, hkArray< hkRefPtr<hkxNode> >& nodes )
{
	if ( !node )
	{
		return;
	}

	nodes.pushBack( node );
	{
		for (int c=0; c < node->m_children.getSize(); ++c)
		{
			findAllNodes( node->m_children[c], nodes);
		}
	}
}

void HK_CALL hkxSceneUtils::findAllMeshNodes(	hkxScene* scene, 
												hkxNode* node, 
												hkArray< hkRefPtr<hkxNode> >& nodes, 
												hkMatrix4* rootTransform,
												hkArray<hkMatrix4>* worldFromLocalTransforms )
{
	if ( !node )
	{
		node = scene->m_rootNode;

		if (!node)
			return; 
	}

	bool computeTransforms = ( rootTransform != HK_NULL ) && ( worldFromLocalTransforms != HK_NULL );

	hkMatrix4 worldFromLocal;
	if ( computeTransforms )
	{
		worldFromLocal.setMul( *rootTransform, node->m_keyFrames[0] );
	}

	hkxMesh* mesh = hkxSceneUtils::getMeshFromNode( node );
	if ( mesh )
	{
		nodes.pushBack( node );

		if ( computeTransforms )
		{
			(*worldFromLocalTransforms).pushBack( worldFromLocal );
		}
	}

	{
		for (int c=0; c < node->m_children.getSize(); ++c)
		{
			findAllMeshNodes(scene, node->m_children[c], nodes, (computeTransforms ? &worldFromLocal : HK_NULL), worldFromLocalTransforms);
		}
	}
}

hkxNode* HK_CALL hkxSceneUtils::findFirstMeshNode(hkxScene* scene)
{
	hkxNode* node = scene->m_rootNode;
	// Go searching for the a node
	for (int i=0;i<node->m_children.getSize();i++)
	{
		hkxNode* subNode = node->m_children[i];
		hkVariant obj = subNode->m_object;

		// Find if its the right kind of node
		if (obj.m_class && (hkString::strCmp(obj.m_class->getName(), hkxMeshClass.getName() ) == 0) )
		{
			return subNode;
		}
	}
	return HK_NULL;
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
