/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Mesh/hkxMeshSectionUtil.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

void hkxBoneIndicesInt8Data::setBoneIndicesDataPtr(const hkxVertexBuffer& vb)
{
	const hkxVertexDescription& vertDecl = vb.getVertexDesc(); 
	const hkxVertexDescription::ElementDecl* boneDecl = vertDecl.getElementDecl(hkxVertexDescription::HKX_DU_BLENDINDICES, 0);

	m_data.m_boneIndicesPtr = (hkUint8*)( vb.getVertexDataPtr(*boneDecl) );
	m_boneIndexStride =  hkUint8(boneDecl->m_byteStride);
}
hkUint16 hkxBoneIndicesInt8Data::getVertexBoneIndex(hkUint32 vertexIdx, hkUint32 influenceIdx)
{
	const hkUint8* vertexBoneIndices = (const hkUint8*)(m_data.m_boneIndicesPtr + vertexIdx*m_boneIndexStride);
	hkUint8 boneIndex = vertexBoneIndices[influenceIdx];

	return hkUint16(boneIndex);
}
void hkxBoneIndicesInt8Data::setVertexBoneIndex(hkUint32 vertexIdx, hkUint32 influenceIdx, hkUint16 boneIndex)
{
	hkUint8* vertexBoneIndices = (hkUint8*)(m_data.m_boneIndicesPtr + vertexIdx*m_boneIndexStride);
	vertexBoneIndices[influenceIdx] = hkUint8(boneIndex);
}

void hkxBoneIndicesInt16Data::setBoneIndicesDataPtr(const hkxVertexBuffer& vb)
{
	const hkxVertexDescription& vertDecl = vb.getVertexDesc(); 
	const hkxVertexDescription::ElementDecl* boneDecl = vertDecl.getElementDecl(hkxVertexDescription::HKX_DU_BLENDINDICES, 0);

	m_data.m_boneIndicesPtr = (hkUint16*)( vb.getVertexDataPtr(*boneDecl) );
	m_boneIndexStride =  hkUint8(boneDecl->m_byteStride/sizeof(hkUint16));
}
hkUint16 hkxBoneIndicesInt16Data::getVertexBoneIndex(hkUint32 vertexIdx, hkUint32 influenceIdx)
{
	const hkUint16* vertexBoneIndices = (const hkUint16*)(m_data.m_boneIndicesPtr + vertexIdx*m_boneIndexStride);
	hkUint16 boneIndex = vertexBoneIndices[influenceIdx];

	return boneIndex;
}
void hkxBoneIndicesInt16Data::setVertexBoneIndex(hkUint32 vertexIdx, hkUint32 influenceIdx, hkUint16 boneIndex)
{
	hkUint16 * vertexBoneIndices = (hkUint16 *)(m_data.m_boneIndicesPtr + vertexIdx*m_boneIndexStride);
	vertexBoneIndices[influenceIdx] = boneIndex;
}

inline void _NormalizeWeights( hkUint8* weights )
{
	hkUint32 total = weights[0] + weights[1] + weights[2] + weights[3];
	hkUint8 qDif = static_cast<hkUint8>( ((255 * 4) - total) / 4 );
	weights[0] = hkUint8(weights[0] + qDif); // get a += for int warning here.. could change to = 
	weights[1] = hkUint8(weights[1] + qDif);
	weights[2] = hkUint8(weights[2] + qDif);
	weights[3] = hkUint8(weights[3] + qDif);
}

void hkxMeshSectionUtil::computeLimitedBoneSection(const hkxMeshSection& si, hkUint32 indexedBoneLimit, 
												   hkxMeshSection& newSection, hkArray<hkBoneRemapping*>& boneMatrixMap )
{
	const hkxVertexBuffer& sourceVertBuf = *si.m_vertexBuffer;
	const hkxVertexDescription& sourceVertDecl = sourceVertBuf.getVertexDesc(); 

	const hkxVertexDescription::ElementDecl* weightDecl = sourceVertDecl.getElementDecl(hkxVertexDescription::HKX_DU_BLENDWEIGHTS, 0);
	const hkxVertexDescription::ElementDecl* boneDecl = sourceVertDecl.getElementDecl(hkxVertexDescription::HKX_DU_BLENDINDICES, 0);
		
	if (!boneDecl || !weightDecl)
	{
		HK_WARN_ALWAYS(0x62feeca, "Skinning information not present, can't compute the limite bone sections");
		return;
	}
	
	
	if(boneDecl->m_type == hkxVertexDescription::HKX_DT_UINT8)
	{
		computeLimitedBoneSection <hkxBoneIndicesInt8Data> (si, indexedBoneLimit, newSection, boneMatrixMap);
	}
	else if(boneDecl->m_type == hkxVertexDescription::HKX_DT_INT16)
	{
		computeLimitedBoneSection<hkxBoneIndicesInt16Data> (si, indexedBoneLimit, newSection, boneMatrixMap);
	}
}


template <class BoneIndicesDataInterface>
void hkxMeshSectionUtil::computeLimitedBoneSection(const hkxMeshSection& si, hkUint32 indexedBoneLimit, 
													   hkxMeshSection& newSection, hkArray<hkBoneRemapping*>& boneMatrixMap)
{
	BoneIndicesDataInterface boneIndicesInterface;

	const hkxVertexBuffer& sourceVertBuf = *si.m_vertexBuffer;
	const hkxVertexDescription& sourceVertDecl = sourceVertBuf.getVertexDesc(); 

	const hkxVertexDescription::ElementDecl* weightDecl = sourceVertDecl.getElementDecl(hkxVertexDescription::HKX_DU_BLENDWEIGHTS, 0);
	int boneWeightStride = weightDecl->m_byteStride;
	const hkUint8* sourceWeights = (hkUint8*)( sourceVertBuf.getVertexDataPtr(*weightDecl) );

	boneIndicesInterface.setBoneIndicesDataPtr(*si.m_vertexBuffer);

    hkLocalArray<hkUint8> consideredBone(boneIndicesInterface.m_maxNumBones);
	consideredBone.setSize(boneIndicesInterface.m_maxNumBones);

	boneMatrixMap.reserve(si.m_indexBuffers.getSize());

	hkArray< hkRefPtr<hkxIndexBuffer> > newIndexBufferArray;
	hkArray<hkUint16> currentTriListSet;

	int maxReferencedBone = -1;

	//
	// Make a list of index buffers (and split some if need be)
	// that can fit inside the bone limit criteria.
	//
	int weightLimit = 0;
	for (int cib=0; cib < si.m_indexBuffers.getSize(); ++cib)
	{
		hkString::memSet((void*)&consideredBone[0], 0, boneIndicesInterface.m_maxNumBones);

		hkBoneRemapping* boneMapping = new hkBoneRemapping; // just an array;
		boneMapping->reserve(indexedBoneLimit);

//		const hkxIndexBuffer* sourceIndexBuffer = si.m_indexBuffers[cib];
		const hkxIndexBuffer& buf = *si.m_indexBuffers[cib];
		bool smallIndices = buf.m_indices16.getSize() > 0;
		HK_ASSERT2( 0x57a67d47, smallIndices || (buf.m_indices32.getSize() > 0), "Mesh must have full index buffers");

		if (buf.m_indexType == hkxIndexBuffer::INDEX_TYPE_TRI_LIST) 
		{
			int numI = buf.m_length;
			currentTriListSet.reserve(numI); 
			currentTriListSet.setSize(0);

			int nt = numI / 3;

			for (int ti=0; ti < nt; ++ti)
			{
				hkUint32 index[3];
				if (smallIndices)
				{
					index[0] = buf.m_indices16[ti*3];
					index[1] = buf.m_indices16[ti*3 + 1];
					index[2] = buf.m_indices16[ti*3 + 2];
				}
				else // assume we have 32 bit ones then.
				{
					index[0] = buf.m_indices32[ti*3];
					index[1] = buf.m_indices32[ti*3 + 1];
					index[2] = buf.m_indices32[ti*3 + 2];
				}

				const hkUint8* weightSet[3]; //XX assumes weight is a uint8
				weightSet[0] = (const hkUint8*)(sourceWeights + index[0]*boneWeightStride );
				weightSet[1] = (const hkUint8*)(sourceWeights + index[1]*boneWeightStride );
				weightSet[2] = (const hkUint8*)(sourceWeights + index[2]*boneWeightStride );

				// for all bones that matter, ie have non zero weights,
				// ( there are potentially 12 bones used, but that would be very rare..)
				bool haveHitLimit= false;
				bool haveHitLimitPreviously;
				do {
					haveHitLimitPreviously = haveHitLimit;
					haveHitLimit = false;
					for (int bs=0; bs < 3; ++bs)
					{
						for (int sbs=0; sbs < 4; ++sbs)
						{
							if ( weightSet[bs][sbs] > weightLimit ) //XX assumes weight is a uint8
							{
								// valid bone to worry about
								hkUint16 boneNum = boneIndicesInterface.getVertexBoneIndex(index[bs], sbs);
								if (!consideredBone[boneNum])
								{
									if (maxReferencedBone < boneNum)
										maxReferencedBone = boneNum;

									if (boneMapping->getSize() == (hkInt32)indexedBoneLimit)
									{	
										// tipped over the edge, batch up what we have for tris
										if ( haveHitLimitPreviously )
										{
											//HK_WARN(0x576b7d47, "Going around in circles, skipping a triangle that can't be accomodated (more bones than there are slots).");
											// fix up the tri set.. increase the weight limit and try again;
											weightLimit += 2; 
											haveHitLimitPreviously = false; // should all be fine now.. or eventually anyway.
										}
										else 
										// make a set with what we have already in the currentTriListSet.
										{
											if (currentTriListSet.getSize() > 0)
											{
												hkxIndexBuffer* newIndexBuffer = new hkxIndexBuffer();
												newIndexBuffer->m_indexType = hkxIndexBuffer::INDEX_TYPE_TRI_LIST;
												newIndexBuffer->m_length = currentTriListSet.getSize();
												newIndexBuffer->m_indices16.setSize( newIndexBuffer->m_length );
												boneMatrixMap.pushBack( boneMapping ); // our current bone mapping
												hkString::memCpy( newIndexBuffer->m_indices16.begin(), currentTriListSet.begin(), currentTriListSet.getSize() * sizeof(hkUint16) );
												newIndexBufferArray.pushBack( newIndexBuffer );
												newIndexBuffer->removeReference();

												boneMapping = new hkBoneRemapping; // just an array;
											}

											// start a new tri list (and reiterate this triangle again)
											currentTriListSet.setSize(0);
											hkString::memSet((void*)&consideredBone[0], 0, boneIndicesInterface.m_maxNumBones);

											boneMapping->setSize(0);
											boneMapping->reserve(indexedBoneLimit);

											weightLimit = 0;
										}

										haveHitLimit = true;	
										break; // either an error tri (too comlex bone connections, caused double batch) or a normal batch and have to just reconsider this tri.									
									}
									else // bone able to be added
									{
										consideredBone[boneNum] = 1;
										boneMapping->pushBack( boneNum ); // make the new matrix pallete mapping
									}
								} // considered bone
							} // if weight > 0
						} // for all sbs

						if (haveHitLimit) 
						{
							break; // end this for loop
						}
					} // for all bs
				} while (haveHitLimit /*&& !haveHitLimitPreviously*/); // while we need to do another trip around on this tri

				// if !errorTri, then the current tri is valid in the current set, so add it and continue:
			//	bool errorTri = haveHitLimit && haveHitLimitPreviously; // double batch on a triangle.
			//	if (!errorTri)
				{
					currentTriListSet.pushBack( (hkUint16) index[0] );
					currentTriListSet.pushBack( (hkUint16) index[1] );
					currentTriListSet.pushBack( (hkUint16) index[2] );
				}
			} // next tri
		}	
		else
		{
			HK_WARN( 0x5a6b7d45, "INDEX_TYPE_TRI_STRIP or FAN not supported in reoder just yet.");
		}

		// if we have comeout of the above, we either have a full (the same index buffer) 
		// or a remainer after the last ibuf create, so need to clean up:
		if (currentTriListSet.getSize() > 0)
		{
			hkxIndexBuffer* newIndexBuffer = new hkxIndexBuffer();
			newIndexBuffer->m_indexType = hkxIndexBuffer::INDEX_TYPE_TRI_LIST;
			newIndexBuffer->m_length = currentTriListSet.getSize();
			newIndexBuffer->m_indices16.setSize( newIndexBuffer->m_length );
			boneMatrixMap.pushBack( boneMapping ); // our current bone mapping
			hkString::memCpy( newIndexBuffer->m_indices16.begin(), currentTriListSet.begin(), currentTriListSet.getSize() * sizeof(hkUint16) );
			newIndexBufferArray.pushBack( newIndexBuffer );
			newIndexBuffer->removeReference();
		}

	} // next orig buffer

	//
	// We now have a list of (possibly new) index buffers and corresponding bone mappings.
	// We need to find vertices that are shared across these index buffers
	// and break that connection as the bone indices will have to change on a per
	// index buffer basis to match the new palletes.
	//
	hkArray<hkUint8> vertCrossIBUsage;
	hkArray<hkUint32> newVertsAdded;

	int numVerts = si.m_vertexBuffer->getNumVertices();
	vertCrossIBUsage.setSize( numVerts );
	hkString::memSet( vertCrossIBUsage.begin(), 0, numVerts * sizeof(hkUint8) );

	int numNewIBuf = newIndexBufferArray.getSize();
	int nib; 
	for (nib = 0; nib < (numNewIBuf-1); ++nib)
	{
		hkxIndexBuffer& buf = *newIndexBufferArray[nib];
		// for each vert in ib, check other ibs > cur ib
		// if that vert is used across IB boundaries:
	
		bool smallIndices = buf.m_indices16.getSize() > 0;

//		const hkxIndexBuffer& buf = ib->getBuffer();
		if (buf.m_indexType == hkxIndexBuffer::INDEX_TYPE_TRI_LIST) 
		{
			unsigned int numI = buf.m_length;
			for (hkUint32 ti=0; ti < numI; ++ti)
			{
				hkUint32 curIndex;
				if (smallIndices)
				{
					curIndex = buf.m_indices16[ti];
				}
				else
				{
					curIndex = buf.m_indices32[ti];
				}

				// check curIndex for membership in all the next index bufs
				for (int nibNext = (nib + 1); nibNext < numNewIBuf; ++nibNext)
				{
					hkxIndexBuffer& otherBuf = *newIndexBufferArray[nibNext];
					bool otherSmallIndices = otherBuf.m_indices16.getSize() > 0;
					if (otherBuf.m_indexType == hkxIndexBuffer::INDEX_TYPE_TRI_LIST) 
					{
						for (hkUint32 ti2=0; ti2 < otherBuf.m_length; ++ti2)
						{
							hkUint32 otherIndex;
							if (otherSmallIndices)
							{
								otherIndex = otherBuf.m_indices16[ti2];
							}
							else
							{
								otherIndex = otherBuf.m_indices32[ti2];
							}

							if (otherIndex == curIndex)
							{
								vertCrossIBUsage[curIndex]++;
								newVertsAdded.pushBack(curIndex); // clone it later (in this order)
								int proposedNewIndex = newVertsAdded.getSize() + numVerts - 1;
								// change the vertindex in anticipation that the vert will be cloned and added again.
								if (otherSmallIndices)
								{
									otherBuf.m_indices16[ti2] = (hkUint16) proposedNewIndex;
								}
								else
								{
									otherBuf.m_indices32[ti2] = (hkUint32) proposedNewIndex;
								}
							}
						}
					}
				}
			}
		}
		// else TRI or FAN, have already warned about that 
	}

	//
	// We have a cross ib usage index and have redone the index bufs,
	// Now we just need a new vert buffer to store any the new verts 
	
	//XXX As we usually have some unused bones in the mix, we remap the verts
	// down anyway and so have to redo the vert buf again anyway.

	int numTotalVerts = sourceVertBuf.getNumVertices() + newVertsAdded.getSize();
	const hkxVertexDescription& srcDesc = sourceVertBuf.getVertexDesc();
	hkxVertexBuffer* newVb = new hkxVertexBuffer();
	hkxVertexBuffer* tempVb = HK_NULL;

	bool changeIndexType = false;
	if ( indexedBoneLimit < 256 )
	{
		const hkxVertexDescription::ElementDecl* boneDecl = srcDesc.getElementDecl(hkxVertexDescription::HKX_DU_BLENDINDICES, 0);
		if ( boneDecl->m_type == hkxVertexDescription::HKX_DT_INT16 )
		{
			changeIndexType = true;
		}
	}

	if ( changeIndexType )
	{
		hkxVertexDescription newDesc;

		int numDecls = srcDesc.m_decls.getSize();
		newDesc.m_decls.setSize( numDecls );

		for ( int i = 0; i < numDecls; ++i )
		{
			const hkxVertexDescription::ElementDecl* srcDecl = srcDesc.getElementDeclByIndex(i);
			hkxVertexDescription::ElementDecl* newDecl = newDesc.getElementDeclByIndex(i);

			newDecl->m_usage = srcDecl->m_usage;
			newDecl->m_numElements = srcDecl->m_numElements;

			if ( newDecl->m_usage != hkxVertexDescription::HKX_DU_BLENDINDICES )
			{
				newDecl->m_type = srcDecl->m_type;
			}
			else
			{
				newDecl->m_type = hkxVertexDescription::HKX_DT_UINT8;
			}
		}

		newVb->setNumVertices( numTotalVerts, newDesc );

		tempVb = new hkxVertexBuffer();
		tempVb->setNumVertices( numTotalVerts, srcDesc );

		// Disable different base type warning
		hkError::getInstance().setEnabled(0xefe34ce, false);
	}
	else
	{
		newVb->setNumVertices( numTotalVerts, srcDesc );
	}

	// copy old verts
	newVb->copy( sourceVertBuf, false );
	if (tempVb)
	{
		tempVb->copy( sourceVertBuf, false );
	}

	// clone other verts
	int newV = sourceVertBuf.getNumVertices();
	for (int vc=0; vc < newVertsAdded.getSize(); ++vc)
	{
		int fromVert = newVertsAdded[vc];
		int destVert = newV + vc;
		newVb->copyVertex( sourceVertBuf, fromVert, destVert );
		if (tempVb)
		{
			tempVb->copyVertex( sourceVertBuf, fromVert, destVert );
		}
	}

	//
	// Finally, run through on a per index buffer basis and 
	// for each index buf used vert, redo the bone index to match the remapped bones:
	// Do each vert once and only once.
	//

	int finalNumVerts = newVb->getNumVertices();
	vertCrossIBUsage.setSize(finalNumVerts);
	hkString::memSet( vertCrossIBUsage.begin(), 0 , finalNumVerts * sizeof(hkUint8) );
	
	hkBoneRemapping inverseBoneRemap;
	inverseBoneRemap.setSize(boneIndicesInterface.m_maxNumBones); // max num bones possible depends on the bone indices type declaration

	const hkxVertexDescription& newVbDesc = newVb->getVertexDesc();
	const hkxVertexDescription::ElementDecl* newBoneWeightDecl = newVbDesc.getElementDecl( hkxVertexDescription::HKX_DU_BLENDWEIGHTS, 0 );
	int newBoneWeightStride = newBoneWeightDecl->m_byteStride;
	hkInt8* weightBase = (hkInt8*)newVb->getVertexDataPtr(*newBoneWeightDecl);

	hkxBoneIndicesInt8Data* boneIndicesInterface8 = HK_NULL;

	if ( changeIndexType )
	{
		boneIndicesInterface8 = new hkxBoneIndicesInt8Data;
		boneIndicesInterface8->setBoneIndicesDataPtr(*newVb);
		boneIndicesInterface.setBoneIndicesDataPtr(*tempVb);
	}
	else
	{
		boneIndicesInterface.setBoneIndicesDataPtr(*newVb);
	}

	for (nib = 0; nib < numNewIBuf; ++nib)
	{
		const hkxIndexBuffer& buf = *newIndexBufferArray[nib];
		bool smallIndices = buf.m_indices16.getSize() > 0;
		// compute inverse mapping
		// any non assigned or weight==0 bones will go to -1 and get a weight of 0 in the end.
		// <todo> Should be changed to store the unmapped bones as just an array of bool rather than reducing the available bones by i
		hkString::memSet( inverseBoneRemap.begin(), -1, boneIndicesInterface.m_maxNumBones * sizeof (hkInt16) );
		hkBoneRemapping* boneRemap = boneMatrixMap[nib];
		for (int bri=0; bri < boneRemap->getSize(); ++bri)
		{
			inverseBoneRemap[ (*boneRemap)[bri] ] = (hkInt16) bri;
		}

		// using the inverse bone remap, set the new indices
		if (buf.m_indexType == hkxIndexBuffer::INDEX_TYPE_TRI_LIST) 
		{
			unsigned int numI = buf.m_length;
			for (hkUint32 ti=0; ti < numI; ++ti)
			{
				hkUint32 curIndex;
				if (smallIndices)
				{
					curIndex = buf.m_indices16[ti];
				}
				else
				{
					curIndex = buf.m_indices32[ti];
				}

				if (vertCrossIBUsage[curIndex] == 0)
				{
					hkUint8* boneWeights = (hkUint8*)( weightBase + curIndex*newBoneWeightStride );  
					
					for(int bic = 0; bic < 4; bic++)
					{
						hkUint16 currBoneIdx = boneIndicesInterface.getVertexBoneIndex(curIndex, bic);
						hkUint16 newBoneIdx = inverseBoneRemap[currBoneIdx];
						if ( boneIndicesInterface8 )
						{
							boneIndicesInterface8->setVertexBoneIndex(curIndex, bic, newBoneIdx);
						}
						else
						{
							boneIndicesInterface.setVertexBoneIndex(curIndex, bic, newBoneIdx);
						}
					}
					
					bool alteredWeights = false;
					for (int bic=0; bic <4; ++bic)
					{
						if ( boneIndicesInterface8 )
						{
							hkUint16 newBoneIdx = boneIndicesInterface8->getVertexBoneIndex(curIndex, bic);
							if ( newBoneIdx == ( boneIndicesInterface8->m_maxNumBones - 1 ) )
							{
								// unmapped bone found (newBoneIdx is (hkUint8)(-1))
								boneIndicesInterface8->setVertexBoneIndex(curIndex, bic, 0);
								boneWeights[bic] = 0;
							}
						}
						else
						{
							hkUint16 newBoneIdx = boneIndicesInterface.getVertexBoneIndex(curIndex, bic);
							if ( newBoneIdx == ( boneIndicesInterface.m_maxNumBones - 1 ) )
							{
								// unmapped bone found (newBoneIdx is (hkUint16)(-1))
								boneIndicesInterface.setVertexBoneIndex(curIndex, bic, 0);
								boneWeights[bic] = 0;
							}
						}
					}
					// <todo> alteredWeights is never set to true
					if (alteredWeights)
					{
						_NormalizeWeights(boneWeights);
					}

					HK_ASSERT( 0x7a6c7d46, 
						boneIndicesInterface8 ?
						( boneIndicesInterface8->getVertexBoneIndex(curIndex, 0) < indexedBoneLimit )
						&& ( boneIndicesInterface8->getVertexBoneIndex(curIndex, 1) < indexedBoneLimit )
						&& ( boneIndicesInterface8->getVertexBoneIndex(curIndex, 2) < indexedBoneLimit )
						&& ( boneIndicesInterface8->getVertexBoneIndex(curIndex, 3) < indexedBoneLimit ) :
						( boneIndicesInterface.getVertexBoneIndex(curIndex, 0) < indexedBoneLimit )
						&& ( boneIndicesInterface.getVertexBoneIndex(curIndex, 1) < indexedBoneLimit )
						&& ( boneIndicesInterface.getVertexBoneIndex(curIndex, 2) < indexedBoneLimit )
						&& ( boneIndicesInterface.getVertexBoneIndex(curIndex, 3) < indexedBoneLimit ));

					vertCrossIBUsage[curIndex] = 1; // used / done
				}
			}
		}
	}

	//
	// Make the new section for it all:
	//
	newSection.m_indexBuffers.swap(newIndexBufferArray);
	newSection.m_vertexBuffer = newVb; 	
	newSection.m_material = si.m_material;

	newVb->removeReference();

	if ( changeIndexType )
	{
		delete boneIndicesInterface8;
		tempVb->removeReference();
		hkError::getInstance().setEnabled(0xefe34ce, true);
	}

	return;
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
