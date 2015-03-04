/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


struct hkWiiCachedBuffer
{	

public:

	hkWiiCachedBuffer() {}

	// This structure represents a data buffer loaded into the Wii's Locked Cache.
	// "size" is the number of bytes in the main memory buffer mirrored to the cache (without pads).
	// "cacheStart" is the index of the location in the 16Kb cache where the buffer (including pads) is placed,
	// and must be a multiple of 32.
	hkWiiCachedBuffer(int size, int cacheStart, u8* cacheBase) : m_size(size), m_cacheStart(cacheStart), m_cacheBase(cacheBase)
	{
		// We allow for up to a 32-byte pad at the start of the buffer, to allow for the requirement of the Locked Cache API 
		// that loads from main memory into the cache must start on a 32-byte aligned address in main memory.
		m_cachedSize = m_size + 32;
		
		m_cacheEnd = m_cacheStart + m_cachedSize - 1;
	}

	HK_FORCE_INLINE int getSize()
	{
		return m_size;
	}
	
	HK_FORCE_INLINE int getStart()
	{
		return m_cacheStart;
	}
	
	HK_FORCE_INLINE int getEnd()
	{
		return m_cacheEnd;
	}
	
	// Start a load from the specified main memory location into the cache. Returns the number of DMA transactions added to the queue.
	HK_FORCE_INLINE int initiateLoad( u8* addressMM )
	{
		m_startMM = addressMM;
		m_startMMaligned = (u8*) HK_NEXT_MULTIPLE_OF(32, hkUlong(m_startMM) - 31);
		m_pad = m_startMM - m_startMMaligned;

		return LCLoadData( m_cacheBase + m_cacheStart, m_startMMaligned, m_cachedSize );
	}
	
	// Given an address in main memory, return the address of the corresponding location in the cache.
	// If the buffer load has completed, using this call we can access the data in main memory location addressMM quickly via the cache.
	HK_FORCE_INLINE u8* getCacheAddress( const u8* addressMM )
	{
		return addressMM - m_startMM + m_cacheBase + m_cacheStart + m_pad;	
	}
	
	// Get the address in the cache of the byte'th byte in the buffer (skipping the pad)
	HK_FORCE_INLINE u8* getCacheAddress( int byte )
	{
		return byte + m_cacheBase + m_cacheStart + m_pad;
	}


private:

	// The address of the buffer in main memory we wish to access in the cache
	u8* m_startMM;

	// The address of the buffer in main memory, aligned to the next lowest 32 byte aligned address (equal to m_startMM if m_startMM is 32 byte aligned)
	u8* m_startMMaligned;
	
	// The pad in bytes needed to move from m_startMMaligned to m_startMM
	u8 m_pad;

	// The size in bytes of the buffer in main memory we pull into the cache, not including pads. 
	int m_size;
	
	// The size in bytes of the buffer in the cache, including pads.
	int m_cachedSize;
	
	// The index of the location in the 16Kb cache where the buffer starts.
	int m_cacheStart;
	
	// The index of the location in the 16Kb cache where the buffer ends (the last byte in the buffer).
	int m_cacheEnd;
	
	// Address of the start of the locked cache
	u8* m_cacheBase;

};


static HK_FORCE_INLINE int _roundup32(int input)
{
 	return HK_NEXT_MULTIPLE_OF(32, input);
}


struct hkWiiLockedCacheInterleaved 
{
	static HK_FORCE_INLINE int computeNumVertsInCache(const hkSkinOperator::Parameters& parameters)
	{	
		// We use the Wii's locked cache to reduce the time to read the input mesh data (positions, normals, starts & influences), by double buffering.
 		// If P & N are interleaved, we divide the 16Kb locked cache into 7 buffers as follows (M=matrices, P=vertices (P&N), S=starts, I=influences)
 		//
	 	//		+---+----+----+----+----+----+----+
	   	//		| M | S0 | P0 | I0 | S1 | P1 | I1 |
		//		+---+----+----+----+----+----+----+
		//
	 	// To construct this, we must first compute numVertsInCache, which means the number of vertices available for reading from the L1 cache at any given time.
	 	// This is the number which fit into buffer 0 or buffer 1 in the cache (0 is used while 1 is loading, and vice versa).
	 	// numVertsInCache is computed to be the maximum number of vertices which can be fitted in.
	 	
 		// After the initial load of the 2 sets of buffers is complete, we do the work of the skinning, double buffering using the locked cache. 
 		// The sequence goes as follows:
	 	// - Wait until only 1 buffer remains to be loaded (completes immediately this time, as 0 buffers loading).
	 	// - Process buffer 0. Initiate a load of new data into buffer 0, and move onto processing buffer 1. 
	 	// - Wait until only 1 buffer remains to be loaded (completes immediately this time, as 1 buffer loading).
	 	// - Process buffer 1. Initiate a load of new data into buffer 1, and move onto processing buffer 0.
	 	// - Wait until only 1 buffer remains to be loaded (stalls now, until buffer 0 load has completed).
	 	// - Process buffer 0. Initiate a load of new data into buffer 0, and move onto processing buffer 1. 
		// - Wait until only 1 buffer remains to be loaded (stalls now, until buffer 1 load has completed).
		// - Process buffer 1. Initiate a load of new data into buffer 1, and move onto processing buffer 0.
		// - (cont...)
		//
		// Assuming Wii characters have about 50 bones, we can fit about 150 vertices into the cache.
	 	// The time to skin 150 vertices will at least partially hide the latency of reading the next 150.
		
		int bytesPerCachedVertex = 0;
		
		// We load enough starts into each buffer for 3 * numVertsInCache, in order to know which influences to fetch for the next loaded buffer
 		bytesPerCachedVertex += 3*sizeof(hkUint16);
 		
 		const hkSkinOperator::SkinStream& inputStream = parameters.m_input;
 		
 		const hkUint32 positionsInStride		= inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_stride;
 		bytesPerCachedVertex += positionsInStride;
 
 		 // Each vertex has between 0 and 4 influences, so to ensure we have all needed we must load numVertsInCache * 4 
 		// (starting from the influence of the first vertex in the buffer)
 		bytesPerCachedVertex += 4*sizeof(hkSkinOperator::BoneInfluence);
 		
 		bytesPerCachedVertex *= 2; // There are two buffers of each type P, N, S, I (for double buffering)
 		
 		// Allow for a 32 byte pad at the start of each buffer, which may be necessary if the location in main memory corresponding to 
	 	// the buffer start is not 32 byte aligned (since the locked cache API requires that we read from a 32 byte aligned address in main memory).
	 	// And also allow for a 32 byte pad at the end of each buffer (since the locked cache API requires that we write to a 32 byte aligned address 
	 	// in the locked cache). Also, according to Nintendo, we should not use the last byte of the locked cached. For safety, we don't use the last 16.
	 	const int size_pad = 7*64 + 16;
	 	
	 	const int size_matrices = parameters.m_numBones * sizeof(hkMatrix4);
 		
 		// Compute the maximum number of vertices we can cache in the available 16Kb, given the constraints.
 		int numVertsInCache = (16384 - size_pad - size_matrices) / ( 2 * bytesPerCachedVertex );
 		
 		return numVertsInCache;
 	}
 	
 	hkWiiLockedCacheInterleaved(int numVertsInCache, const hkSkinOperator::Parameters& parameters) : m_numVertsInCache(numVertsInCache), m_parameters(parameters)
 	{	
		u8* lockedCacheBase = (u8*)LCGetBase();

		const hkSkinOperator::SkinStream& inputStream = parameters.m_input;
		const hkUint32 positionsInStride = inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_stride;
		
		const int mBufferSize = m_parameters.m_numBones * sizeof(hkMatrix4);
		const int sBufferSize = (3 * m_numVertsInCache) * sizeof(hkUint16);
		const int vBufferSize = m_numVertsInCache * positionsInStride;
		const int iBufferSize = m_numVertsInCache * 4 * sizeof(hkSkinOperator::BoneInfluence);
	
		hkWiiCachedBuffer mBuffer(mBufferSize, 0, lockedCacheBase); 
	 	hkWiiCachedBuffer sBuffer0(sBufferSize, _roundup32(mBuffer.getEnd() +1), lockedCacheBase);
	 	hkWiiCachedBuffer vBuffer0(vBufferSize, _roundup32(sBuffer0.getEnd()+1), lockedCacheBase);
	 	hkWiiCachedBuffer iBuffer0(iBufferSize, _roundup32(vBuffer0.getEnd()+1), lockedCacheBase);
	 	hkWiiCachedBuffer sBuffer1(sBufferSize, _roundup32(iBuffer0.getEnd()+1), lockedCacheBase);
	 	hkWiiCachedBuffer vBuffer1(vBufferSize, _roundup32(sBuffer1.getEnd()+1), lockedCacheBase);
	 	hkWiiCachedBuffer iBuffer1(iBufferSize, _roundup32(vBuffer1.getEnd()+1), lockedCacheBase);
	
		m_matrices = mBuffer;
		m_startBuffer[0] = sBuffer0;
		m_vertexBuffer[0] = vBuffer0;
		m_influenceBuffer[0] = iBuffer0;
		m_startBuffer[1] = sBuffer1;
		m_vertexBuffer[1] = vBuffer1;
		m_influenceBuffer[1] = iBuffer1;
	}
	
	HK_FORCE_INLINE int getNumVertsInCache()
	{
		return m_numVertsInCache;
	}
	
	const hkSkinOperator::Parameters& m_parameters;
	int m_numVertsInCache;
	
	// Buffers
	hkWiiCachedBuffer m_matrices;
	hkWiiCachedBuffer m_startBuffer[2];
	hkWiiCachedBuffer m_vertexBuffer[2];
	hkWiiCachedBuffer m_influenceBuffer[2];
	
	HK_FORCE_INLINE hkWiiCachedBuffer& getMatrixBuffer()
	{
		return m_matrices;
	}
	
	HK_FORCE_INLINE hkWiiCachedBuffer& getStartsBuffer(int b)
	{
		return m_startBuffer[b];
	}
	
	HK_FORCE_INLINE hkWiiCachedBuffer& getPositionsBuffer(int b)
	{
		return m_vertexBuffer[b];
	}
	
	HK_FORCE_INLINE hkWiiCachedBuffer& getNormalsBuffer(int b)
	{
		return m_vertexBuffer[b];
	}
	
	HK_FORCE_INLINE hkWiiCachedBuffer& getInfluencesBuffer(int b)
	{
		return m_influenceBuffer[b];
	}
	
	// Access to values
	HK_FORCE_INLINE const hkMatrix4& getMatrix(int boneIndex)
	{
		return *reinterpret_cast<const hkMatrix4*>( m_matrices.getCacheAddress(boneIndex * sizeof(hkMatrix4)) );
	}
	
	HK_FORCE_INLINE void getOriginalPosition(int b, const u8* position, hkVector4& positionOut)
	{
		positionOut = *reinterpret_cast<const hkVector4*> ( m_vertexBuffer[b].getCacheAddress((u8*)position) );
	}
	
	HK_FORCE_INLINE void getOriginalNormal(int b, const u8* normal, hkVector4& normalOut) 
	{ 
		normalOut = *reinterpret_cast<const hkVector4*> ( m_vertexBuffer[b].getCacheAddress((u8*)normal) );
	}
	
	HK_FORCE_INLINE hkUint16 getStart(int b, int vertexIndex)
	{
		u8* start = (u8*)m_parameters.m_startInfluencePerVertex + vertexIndex * sizeof(hkUint16);
		return *reinterpret_cast<hkUint16*>( m_startBuffer[b].getCacheAddress(start) );
	}
	
	HK_FORCE_INLINE const hkSkinOperator::BoneInfluence& getInfluence(int b, const hkSkinOperator::BoneInfluence* currentInfluence)
	{
		return *reinterpret_cast<const hkSkinOperator::BoneInfluence*>( m_influenceBuffer[b].getCacheAddress((u8*)currentInfluence));
	}
	
	// Initiate loads
	HK_FORCE_INLINE int loadMatrices()
	{
		return m_matrices.initiateLoad((u8*)m_parameters.m_compositeMatrices);
	}
	
	HK_FORCE_INLINE int loadStarts(int b, int startVertexIndex)
	{
		u8* start = (u8*)m_parameters.m_startInfluencePerVertex + startVertexIndex * sizeof(hkUint16);
		return m_startBuffer[b].initiateLoad(start);
	}
	
	HK_FORCE_INLINE int loadInfluences(int b, int startInfluenceIndex)
	{
		u8* influence = (u8*)m_parameters.m_boneInfluences + startInfluenceIndex * sizeof(hkSkinOperator::BoneInfluence);
		return m_influenceBuffer[b].initiateLoad(influence);
	}
	
	HK_FORCE_INLINE int loadPositions(int b, int startVertexIndex)
	{
		const hkSkinOperator::SkinStream& inputStream = m_parameters.m_input;
		const hkUint32 positionsInStride = inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_stride;
		
		u8* position = (u8*)hkAddByteOffsetConst(inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_start,
							inputStream.m_positions.m_offset + positionsInStride * startVertexIndex);
		
		return m_vertexBuffer[b].initiateLoad(position);
	}
	
	HK_FORCE_INLINE int loadNormals(int b, int startVertexIndex)
	{
		return 0;
	}
};	


struct hkWiiLockedCacheNonInterleaved 
{
	static HK_FORCE_INLINE int computeNumVertsInCache(const hkSkinOperator::Parameters& parameters)
	{
		// We use the Wii's locked cache to reduce the time to read the input mesh data (positions, normals, starts & influences), by double buffering.
 		// If P & N are non-interleaved, we divide the 16Kb locked cache into 9 buffers as follows (M=matrices, P=positions, N=normals, S=starts, I=influences)
 		//
		//		+---+----+----+----+----+----+----+----+----+
	   	//		| M | S0 | P0 | N0 | I0 | S1 | P1 | N1 | I1 |
		//		+---+----+----+----+----+----+----+----+----+
		//
	 	// To construct this, we must first compute numVertsInCache, which means the number of vertices available for reading from the L1 cache at any given time.
	 	// This is the number which fit into buffer 0 or buffer 1 in the cache (0 is used while 1 is loading, and vice versa).
	 	// numVertsInCache is computed to be the maximum number of vertices which can be fitted in.
		
		int bytesPerCachedVertex = 0;
		
		// We load enough starts into each buffer for 3 * numVertsInCache, in order to know which influences to fetch for the next loaded buffer
 		bytesPerCachedVertex += 3*sizeof(hkUint16);
 		
 		const hkSkinOperator::SkinStream& inputStream = parameters.m_input;
 		
 		const hkUint32 positionsInStride = inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_stride;
		const hkUint32 normalsInStride = inputStream.m_buffers[inputStream.m_normals.m_bufferIndex].m_stride;
 		bytesPerCachedVertex += (positionsInStride + normalsInStride);
 
 		 // Each vertex has between 0 and 4 influences, so to ensure we have all needed we must load numVertsInCache * 4 
 		// (starting from the influence of the first vertex in the buffer)
 		bytesPerCachedVertex += 4*sizeof(hkSkinOperator::BoneInfluence);
 		
 		bytesPerCachedVertex *= 2; // There are two buffers of each type P, N, S, I (for double buffering)
 		
 		// Allow for a 32 byte pad at the start of each buffer, which may be necessary if the location in main memory corresponding to 
	 	// the buffer start is not 32 byte aligned (since the locked cache API requires that we read from a 32 byte aligned address in main memory).
	 	// And also allow for a 32 byte pad at the end of each buffer (since the locked cache API requires that we write to a 32 byte aligned address 
	 	// in the locked cache). Also, according to Nintendo, we should not use the last byte of the locked cached. For safety, we don't use the last 16.
	 	const int size_pad = 9*64 + 16;
	 	
	 	const int size_matrices = parameters.m_numBones * sizeof(hkMatrix4);
 		
 		// Compute the maximum number of vertices we can cache in the available 16Kb, given the constraints.
 		int numVertsInCache = (16384 - size_pad - size_matrices) / ( 2 * bytesPerCachedVertex );
 		
 		return numVertsInCache;
 	}
 	
 	hkWiiLockedCacheNonInterleaved(int numVertsInCache, const hkSkinOperator::Parameters& parameters) : m_numVertsInCache(numVertsInCache), m_parameters(parameters)
 	{	
		u8* lockedCacheBase = (u8*)LCGetBase();
		
		const hkSkinOperator::SkinStream& inputStream = parameters.m_input;

		const hkUint32 positionsInStride = inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_stride;
		const hkUint32 normalsInStride = inputStream.m_buffers[inputStream.m_normals.m_bufferIndex].m_stride;
		
		const int mBufferSize = m_parameters.m_numBones * sizeof(hkMatrix4);
		const int sBufferSize = (3 * m_numVertsInCache) * sizeof(hkUint16);
		const int pBufferSize = m_numVertsInCache * positionsInStride;
		const int nBufferSize = m_numVertsInCache * normalsInStride;
		const int iBufferSize = m_numVertsInCache * 4 * sizeof(hkSkinOperator::BoneInfluence);
		
		hkWiiCachedBuffer mBuffer(mBufferSize, 0, lockedCacheBase); 
	 	hkWiiCachedBuffer sBuffer0(sBufferSize, _roundup32(mBuffer.getEnd() +1), lockedCacheBase);
	 	hkWiiCachedBuffer pBuffer0(pBufferSize, _roundup32(sBuffer0.getEnd()+1), lockedCacheBase);
	 	hkWiiCachedBuffer nBuffer0(nBufferSize, _roundup32(pBuffer0.getEnd()+1), lockedCacheBase);
	 	hkWiiCachedBuffer iBuffer0(iBufferSize, _roundup32(nBuffer0.getEnd()+1), lockedCacheBase);
	 	hkWiiCachedBuffer sBuffer1(sBufferSize, _roundup32(iBuffer0.getEnd()+1), lockedCacheBase);
	 	hkWiiCachedBuffer pBuffer1(pBufferSize, _roundup32(sBuffer1.getEnd()+1), lockedCacheBase);
	 	hkWiiCachedBuffer nBuffer1(nBufferSize, _roundup32(pBuffer1.getEnd()+1), lockedCacheBase);
	 	hkWiiCachedBuffer iBuffer1(iBufferSize, _roundup32(nBuffer1.getEnd()+1), lockedCacheBase);
	
		m_matrices = mBuffer;
		m_startBuffer[0] = sBuffer0;
		m_positionsBuffer[0] = pBuffer0;
		m_normalsBuffer[0] = nBuffer0;
		m_influenceBuffer[0] = iBuffer0;
		m_startBuffer[1] = sBuffer1;
		m_positionsBuffer[1] = pBuffer1;
		m_normalsBuffer[1] = nBuffer1;
		m_influenceBuffer[1] = iBuffer1;
	}
	
	HK_FORCE_INLINE int getNumVertsInCache()
	{
		return m_numVertsInCache;
	}
	
	const hkSkinOperator::Parameters& m_parameters;
	int m_numVertsInCache;
	
	// Buffers	
	hkWiiCachedBuffer m_matrices;
	hkWiiCachedBuffer m_startBuffer[2];
	hkWiiCachedBuffer m_positionsBuffer[2];
	hkWiiCachedBuffer m_normalsBuffer[2];
	hkWiiCachedBuffer m_influenceBuffer[2];

	HK_FORCE_INLINE hkWiiCachedBuffer& getMatrixBuffer()
	{
		return m_matrices;
	}
	
	HK_FORCE_INLINE hkWiiCachedBuffer& getStartsBuffer(int b)
	{
		return m_startBuffer[b];
	}
	
	HK_FORCE_INLINE hkWiiCachedBuffer& getPositionsBuffer(int b)
	{
		return m_positionsBuffer[b];
	}
	
	HK_FORCE_INLINE hkWiiCachedBuffer& getNormalsBuffer(int b)
	{
		return m_normalsBuffer[b];
	}
	
	HK_FORCE_INLINE hkWiiCachedBuffer& getInfluencesBuffer(int b)
	{
		return m_influenceBuffer[b];
	}
	
	// Access to values
	HK_FORCE_INLINE const hkMatrix4& getMatrix(int boneIndex)
	{
		return *reinterpret_cast<const hkMatrix4*>( m_matrices.getCacheAddress(boneIndex * sizeof(hkMatrix4)) );
	}
	
	HK_FORCE_INLINE void getOriginalPosition(int b, const u8* position, hkVector4& positionOut)
	{
		positionOut = *reinterpret_cast<const hkVector4*> ( m_positionsBuffer[b].getCacheAddress(position) );
	}
	
	HK_FORCE_INLINE void getOriginalNormal(int b, const u8* normal, hkVector4& normalOut)
	{
		normalOut = *reinterpret_cast<const hkVector4*> ( m_normalsBuffer[b].getCacheAddress(normal) );
	}
	
	HK_FORCE_INLINE hkUint16 getStart(int b, int vertexIndex)
	{
		u8* start = (u8*)m_parameters.m_startInfluencePerVertex + vertexIndex * sizeof(hkUint16);
		return *reinterpret_cast<hkUint16*>( m_startBuffer[b].getCacheAddress(start) );
	}
	
	HK_FORCE_INLINE const hkSkinOperator::BoneInfluence& getInfluence(int b, const hkSkinOperator::BoneInfluence* currentInfluence)
	{
		return *reinterpret_cast<const hkSkinOperator::BoneInfluence*>( m_influenceBuffer[b].getCacheAddress((u8*)currentInfluence));
	}
	
	// Initiate loads
	HK_FORCE_INLINE int loadMatrices()
	{
		return m_matrices.initiateLoad((u8*)m_parameters.m_compositeMatrices);
	}
	
	HK_FORCE_INLINE int loadStarts(int b, int startVertexIndex)
	{
		u8* start = (u8*)m_parameters.m_startInfluencePerVertex + startVertexIndex * sizeof(hkUint16);
		return m_startBuffer[b].initiateLoad(start);
	}
	
	HK_FORCE_INLINE int loadInfluences(int b, int startInfluenceIndex)
	{
		u8* influence = (u8*)m_parameters.m_boneInfluences + startInfluenceIndex * sizeof(hkSkinOperator::BoneInfluence);
		return m_influenceBuffer[b].initiateLoad(influence);
	}
	
	HK_FORCE_INLINE int loadPositions(int b, int startVertexIndex)
	{
		const hkSkinOperator::SkinStream& inputStream = m_parameters.m_input;
		const hkUint32 positionsInStride = inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_stride;
		
		u8* position = (u8*)hkAddByteOffsetConst(inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_start,
							inputStream.m_positions.m_offset + positionsInStride * startVertexIndex);
		
		return m_positionsBuffer[b].initiateLoad(position);
	}
	
	HK_FORCE_INLINE int loadNormals(int b, int startVertexIndex)
	{
		const hkSkinOperator::SkinStream& inputStream = m_parameters.m_input;
		const hkUint32 normalsInStride = inputStream.m_buffers[inputStream.m_normals.m_bufferIndex].m_stride;
		
		u8* normal = (u8*)hkAddByteOffsetConst(inputStream.m_buffers[inputStream.m_normals.m_bufferIndex].m_start,
						inputStream.m_normals.m_offset + normalsInStride * startVertexIndex);
		
		return m_normalsBuffer[b].initiateLoad(normal);
	}

};	


struct hkWiiWriteToPipe
{
	HK_FORCE_INLINE void initPipe(u8* positionsOut, int numVertices, int vertexStride)
	{
		DCInvalidateRange(positionsOut, numVertices * vertexStride);
		m_writeGatherRegister = GXRedirectWriteGatherPipe( positionsOut );
		m_pipeValid = true;
	}

	HK_FORCE_INLINE void closePipe()
	{
		GXRestoreWriteGatherPipe();
	}

	HK_FORCE_INLINE void redirectPipe(u8* positionsOut)
	{
		if (!m_pipeValid)
		{
			GXRestoreWriteGatherPipe();
			m_writeGatherRegister = GXRedirectWriteGatherPipe( positionsOut );
			m_pipeValid = true;	
		}
	}
	
	HK_FORCE_INLINE void invalidatePipe()
	{
		m_pipeValid = false;
	}
	
	HK_FORCE_INLINE void writePosition(u8* positionsOut, const hkVector4& position)
	{
		*(volatile float*)m_writeGatherRegister = position(0);
		*(volatile float*)m_writeGatherRegister = position(1);
		*(volatile float*)m_writeGatherRegister = position(2);
		*(volatile float*)m_writeGatherRegister = position(3);
	}
	
	HK_FORCE_INLINE void writeNormal(u8* normalsOut, const hkVector4& normal)
	{
		*(volatile float*)m_writeGatherRegister = normal(0);
		*(volatile float*)m_writeGatherRegister = normal(1);
		*(volatile float*)m_writeGatherRegister = normal(2);
		*(volatile float*)m_writeGatherRegister = normal(3);
	}

	bool m_pipeValid;
	volatile void* m_writeGatherRegister;
};

struct hkWiiWriteToMemory
{
	HK_FORCE_INLINE void initPipe(u8* positionsOut, int numVertices, int vertexStride) {}
	HK_FORCE_INLINE void closePipe() {}
	HK_FORCE_INLINE void redirectPipe(u8* positionsOut) {}
	HK_FORCE_INLINE void invalidatePipe() {}

	HK_FORCE_INLINE void writePosition(u8* positionsOut, const hkVector4& position)
	{
		*(reinterpret_cast<hkVector4*>(positionsOut)) = position;
	}
	
	HK_FORCE_INLINE void writeNormal(u8* normalsOut, const hkVector4& normal)
	{
		*(reinterpret_cast<hkVector4*>(normalsOut)) = normal;
	}
};


template < class LockedCache, class WriteMethod >
void hkSkinWii_PN(const hkSkinOperator::Parameters& parameters, LockedCache& lockedCache, WriteMethod& writeMethod)
{
	const hkMatrix4* compositeMatrices = parameters.m_compositeMatrices;
	const hkSkinOperator::BoneInfluence* boneInfluences = parameters.m_boneInfluences;
	const hkUint16* startInfluencePerVertex = parameters.m_startInfluencePerVertex;
	
	const hkSkinOperator::SkinStream& inputStream = parameters.m_input;
	const hkSkinOperator::SkinStream& outputStream = parameters.m_output;
	
	const hkUint32 numVertices = parameters.m_numVertices;

	const hkReal oneOver255 = 3.921568627e-03f;
 
	const u8* positionsIn	= (u8*)hkAddByteOffsetConst(inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_start, inputStream.m_positions.m_offset);
	u8* positionsOut		= (u8*)hkAddByteOffset(outputStream.m_buffers[outputStream.m_positions.m_bufferIndex].m_start, outputStream.m_positions.m_offset);
	const hkUint32 positionsInStride	= inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_stride;
	const hkUint32 positionsOutStride	= outputStream.m_buffers[outputStream.m_positions.m_bufferIndex].m_stride;
	
	const u8* normalsIn	= (u8*)hkAddByteOffsetConst(inputStream.m_buffers[inputStream.m_normals.m_bufferIndex].m_start, inputStream.m_normals.m_offset);
	u8* normalsOut		= (u8*)hkAddByteOffset(outputStream.m_buffers[outputStream.m_normals.m_bufferIndex].m_start, outputStream.m_normals.m_offset);
	const hkUint32 normalsInStride	= inputStream.m_buffers[inputStream.m_normals.m_bufferIndex].m_stride;
	const hkUint32 normalsOutStride	= outputStream.m_buffers[outputStream.m_normals.m_bufferIndex].m_stride;

	int numBones = parameters.m_numBones;
	int numVertsInCache = lockedCache.getNumVertsInCache();
	
	OSInitFastCast();
		
	// Set up the Write-Gather Pipe, if enabled
	writeMethod.initPipe(positionsOut, numVertices, positionsOutStride);
	
	// Load the composite matrices 
 	lockedCache.loadMatrices();
 	
 	// Do the initial load of the starts
 	int numTransactions_S = 0;
 	for (int b=0; b<2; ++b)
	{
		numTransactions_S += lockedCache.loadStarts(b, b*numVertsInCache);
	}
	LCQueueWait(0);
	
	// Do the initial load of the vertices & influences
 	int numTransactions_V = 0;
 	int numTransactions_I = 0;
 	for (int b=0; b<2; ++b)
	{
		numTransactions_V  = lockedCache.loadPositions(b, b*numVertsInCache);
		numTransactions_V += lockedCache.loadNormals(b, b*numVertsInCache);
	
		const hkUint16 startInfluence = lockedCache.getStart(b, b*numVertsInCache);
		numTransactions_I = lockedCache.loadInfluences(b, startInfluence);
	}
	LCQueueWait(numTransactions_V + numTransactions_I);

 	// The total number of DMA transactions needed to load 1 buffer (of each type: starts, vertices & influences)
 	int numTransactions = numTransactions_S + numTransactions_V + numTransactions_I;
	
	int currentVertex = 0;
	int buffersDone = 0;
	
	while (currentVertex < numVertices)
	{
		int b = (buffersDone % 2);

		// Wait until only 1 buffer (of each type: starts, vertices & influences) remains to be loaded into the cache
		LCQueueWait(numTransactions);
		
		for (int i=0; i<numVertsInCache && currentVertex<numVertices; ++i)
		{	
			// Get vertex influence start of the current vertex
			const hkUint16 startInfluence = lockedCache.getStart(b, currentVertex);
			const hkUint16 endInfluence = lockedCache.getStart(b, currentVertex+1);
			const int numInfluences = endInfluence - startInfluence;
			
			if (numInfluences>0)
			{
				writeMethod.redirectPipe(positionsOut);

				const hkSkinOperator::BoneInfluence* currentInfluence = boneInfluences + startInfluence;
			
				hkVector4 originalPosition, originalNormal;
				lockedCache.getOriginalPosition(b, positionsIn, originalPosition);
				lockedCache.getOriginalNormal(b, normalsIn, originalNormal);

				hkVector4 blendedPosition, blendedNormal;
				blendedPosition.setZero();
				blendedNormal.setZero();
				
				for (int inf=0; inf<numInfluences; inf++)
				{
					hkVector4 transformedPosition, transformedNormal;
					
					const hkSkinOperator::BoneInfluence& cachedInfluence = lockedCache.getInfluence(b, currentInfluence);
					const hkUint8 weightInt = cachedInfluence.m_weight;
				
					const hkMatrix4& boneTransform = lockedCache.getMatrix(cachedInfluence.m_boneIndex);
					
					hkReal weightReal;
					OSu8tof32(const_cast<hkUint8*>(&weightInt), &weightReal);
					weightReal *= oneOver255;
					
					boneTransform.transformPosition (originalPosition, transformedPosition);
					boneTransform.transformDirection(originalNormal, transformedNormal);

					const hkSimdReal weightSimdReal = hkSimdReal::convert(weightReal);
					blendedPosition.addMul(weightSimdReal, transformedPosition);
					blendedNormal.addMul(weightSimdReal, transformedNormal);
					
					currentInfluence++;
				}
				
				writeMethod.writePosition(positionsOut, blendedPosition);
				writeMethod.writeNormal(normalsOut, blendedNormal);
			}
			
			else
			{
				// The next vertex stored is not contiguous to the last. So have to redirect the write-gather pipe (if enabled)
				writeMethod.invalidatePipe();
			}
			
			// Next!
			positionsIn = hkAddByteOffsetConst (positionsIn, positionsInStride);
			normalsIn = hkAddByteOffsetConst (normalsIn, normalsInStride);
			
			positionsOut = hkAddByteOffset (positionsOut, positionsOutStride);
			normalsOut = hkAddByteOffset (normalsOut, normalsOutStride);
		
			currentVertex++;
		}
		
		// Start loading the next buffer of starts, vertices and influences
		const int nextBufferVertex = (buffersDone + 2) * numVertsInCache;
		const hkUint16 nextBufferStartInfluence = lockedCache.getStart(b, nextBufferVertex);

		numTransactions  = lockedCache.loadInfluences(b, nextBufferStartInfluence);
		numTransactions += lockedCache.loadStarts(b, nextBufferVertex);
		numTransactions += lockedCache.loadPositions(b, nextBufferVertex);
		numTransactions += lockedCache.loadNormals(b, nextBufferVertex);

		buffersDone++;
	}

	writeMethod.closePipe();
}	



static void hkSkinWii_P(const hkSkinOperator::Parameters& parameters, hkWiiLockedCacheInterleaved& lockedCache, hkWiiWriteToMemory& writeMethod)
{
	const hkMatrix4* compositeMatrices = parameters.m_compositeMatrices;
	const hkSkinOperator::BoneInfluence* boneInfluences = parameters.m_boneInfluences;

	const hkUint32 numVertices = parameters.m_numVertices;

	const hkReal oneOver255 = 3.921568627e-03f;
	
	const hkSkinOperator::SkinStream& inputStream = parameters.m_input;
	const hkSkinOperator::SkinStream& outputStream = parameters.m_output;
	
	const u8* positionsIn 	= (u8*)hkAddByteOffsetConst(inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_start, inputStream.m_positions.m_offset);
	u8*		positionsOut 	= (u8*)hkAddByteOffset(outputStream.m_buffers[outputStream.m_positions.m_bufferIndex].m_start, outputStream.m_positions.m_offset);
	const hkUint32 positionsInStride	= inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_stride;
	const hkUint32 positionsOutStride 	= outputStream.m_buffers[outputStream.m_positions.m_bufferIndex].m_stride;
	

	int numBones = parameters.m_numBones;
	int numVertsInCache = lockedCache.getNumVertsInCache();
	
	OSInitFastCast();
	
	// Load the composite matrices 
 	lockedCache.loadMatrices();

 	// Do the initial load of the starts
 	int numTransactions_S = 0;
 	for (int b=0; b<2; ++b)
	{
		numTransactions_S += lockedCache.loadStarts(b, b*numVertsInCache);
	}
	LCQueueWait(0);
	
	// Do the initial load of the vertices & influences
 	int numTransactions_V = 0;
 	int numTransactions_I = 0;
 	for (int b=0; b<2; ++b)
	{
		numTransactions_V = lockedCache.loadPositions(b, b*numVertsInCache);

		const hkUint16 startInfluence = lockedCache.getStart(b, b*numVertsInCache);
		numTransactions_I = lockedCache.loadInfluences(b, startInfluence);
	}
 	LCQueueWait(numTransactions_V + numTransactions_I);
 	
 	// The total number of DMA transactions needed to load 1 of each of a starts (S), vertex (V), and influence (I) buffer.
 	int numTransactions = numTransactions_S + numTransactions_V + numTransactions_I;
	
	int currentVertex = 0;
	int buffersDone = 0;

	while (currentVertex < numVertices)
	{
		int b = (buffersDone % 2);
		
		// Wait until only 1 buffer (of each type: starts, vertices & influences) remains to be loaded into the cache
		LCQueueWait(numTransactions);

		for (int i=0; i<numVertsInCache && currentVertex<numVertices; ++i)
		{	
			// Get vertex influence start of the current vertex
			const hkUint16 startInfluence = lockedCache.getStart(b, currentVertex);
			const hkUint16 endInfluence = lockedCache.getStart(b, currentVertex+1);
			const int numInfluences = endInfluence - startInfluence;
			
			if (numInfluences>0)
			{
				const hkSkinOperator::BoneInfluence* currentInfluence = boneInfluences + startInfluence;
			
				hkVector4 originalPosition;
				lockedCache.getOriginalPosition(b, positionsIn, originalPosition);
			
				hkVector4 blendedPosition;
				blendedPosition.setZero();
			
				for (int inf=0; inf<numInfluences; inf++)
				{
					hkVector4 transformedPosition, transformedNormal;
					
					const hkSkinOperator::BoneInfluence& cachedInfluence = lockedCache.getInfluence(b, currentInfluence);
					const hkUint8 weightInt = cachedInfluence.m_weight;
		
					const hkMatrix4& boneTransform = lockedCache.getMatrix(cachedInfluence.m_boneIndex);
					
					hkReal weightReal;
					OSu8tof32(const_cast<hkUint8*>(&weightInt), &weightReal);
					weightReal *= oneOver255;
					
					boneTransform.transformPosition (originalPosition, transformedPosition);
					
					blendedPosition.addMul(hkSimdReal::convert(weightReal), transformedPosition);

					currentInfluence++;
				}
				
				writeMethod.writePosition(positionsOut, blendedPosition);
			}
			
			// Next!
			positionsIn = hkAddByteOffsetConst (positionsIn, positionsInStride);
			positionsOut = hkAddByteOffset (positionsOut, positionsOutStride);

			currentVertex++;
		}
		
		// Start loading the next buffer of starts, vertices and influences
		const int nextBufferVertex = (buffersDone + 2) * numVertsInCache;
		const hkUint16 nextBufferStartInfluence = lockedCache.getStart(b, nextBufferVertex);

		numTransactions  = lockedCache.loadInfluences(b, nextBufferStartInfluence);
		numTransactions += lockedCache.loadStarts(b, nextBufferVertex);
		numTransactions += lockedCache.loadPositions(b, nextBufferVertex);

		buffersDone++;
	}
}	
	
	
static bool hkWiiSkinning_PN(const hkSkinOperator::Parameters& parameters)
{
	const hkUint32 numVertices = parameters.m_numVertices;
	
	const hkSkinOperator::SkinStream& inputStream = parameters.m_input;
	const hkSkinOperator::SkinStream& outputStream = parameters.m_output;

	const void*  positionsIn		= hkAddByteOffsetConst(inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_start, inputStream.m_positions.m_offset);
	void*		 positionsOut	= hkAddByteOffset(outputStream.m_buffers[outputStream.m_positions.m_bufferIndex].m_start, outputStream.m_positions.m_offset);
	const hkUint32 positionsInStride	= inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_stride;
	const hkUint32 positionsOutStride	= outputStream.m_buffers[outputStream.m_positions.m_bufferIndex].m_stride;
	
	const void*  normalsIn	= hkAddByteOffsetConst(inputStream.m_buffers[inputStream.m_normals.m_bufferIndex].m_start, inputStream.m_normals.m_offset);
	void*		 normalsOut	= hkAddByteOffset(outputStream.m_buffers[outputStream.m_normals.m_bufferIndex].m_start, outputStream.m_normals.m_offset);
	const hkUint32 normalsInStride	= inputStream.m_buffers[inputStream.m_normals.m_bufferIndex].m_stride;
	const hkUint32 normalsOutStride	= outputStream.m_buffers[outputStream.m_normals.m_bufferIndex].m_stride;

	// We take PN as interleaved if and only if the first normal is bracketed in memory between the first and second positions,
	// otherwise the positions and normals will be cached separately. Note that the atypical case in which PN are interleaved, but
	// the normals come first, will be treated as non-interleaved in this scheme.
	bool interleavedInputPN = ( normalsIn > (u8*)positionsIn ) && ( normalsIn < (u8*)positionsIn + positionsInStride );
	
	// In our HKG vertex buffers on Wii, the positions and normals (16 bytes per P or N) are interleaved separately from the colors and UVs.
	// So each 32-byte-aligned 32-byte-length PN pair can be written quickly via the Write-Gather Pipe.	
	// We support this currently for customers only if their output vertex buffers on Wii are also of this precise form, i.e.
	// only if the output buffer positions and normals are interleaved together like PNPNPN.., where the first P is 32-byte-aligned,
	// and each P and N is 16 bytes in length
	bool aligned32OutputP = ((void*)HK_NEXT_MULTIPLE_OF(32, hkUlong(positionsOut) - 31) == positionsOut);
	bool interleavedOutputPN = ((u8*)positionsOut + 16 == (u8*)normalsOut) && (positionsOutStride == 32) && (normalsOutStride == 32);
	bool writeGatherEnabled = (aligned32OutputP && interleavedOutputPN);
	
	// Flush all locations we will (potentially) need to read from the cache, to make sure they are read fresh from main memory.
	DCFlushRange((u8*)parameters.m_compositeMatrices,       parameters.m_numBones * sizeof(hkMatrix4));
	DCFlushRange((u8*)parameters.m_boneInfluences,          numVertices * 4 * sizeof(hkSkinOperator::BoneInfluence));
	DCFlushRange((u8*)parameters.m_startInfluencePerVertex, numVertices * sizeof(hkUint16));
	DCFlushRange((u8*)positionsIn, numVertices * positionsInStride);
	
	// We assume skinning via the locked cache is not likely to be fast if less than MIN_NUM_CACHED_VERTS fit in the cache.
	const int MIN_NUM_CACHED_VERTS = 16;
	
	// Do skinning
	if (interleavedInputPN)
	{
		int numVertsInCache = hkWiiLockedCacheInterleaved::computeNumVertsInCache(parameters);
		if (numVertsInCache<MIN_NUM_CACHED_VERTS)
		{
			 HK_WARN_ONCE (0x1017a38d, "Cannot fit interleaved input buffer with stride " << positionsInStride << \
			 			" bytes in the 16Kb locked cache, therefore reverting to generic skinning for this buffer" );
			 return false;
		}
		
		hkWiiLockedCacheInterleaved lockedCache(numVertsInCache, parameters);	
	
		if (writeGatherEnabled)
		{
			hkWiiWriteToPipe writeMethod;
			
			hkSkinWii_PN(parameters, lockedCache, writeMethod);
		}
		else
		{	
			hkWiiWriteToMemory writeMethod;
			
			hkSkinWii_PN(parameters, lockedCache, writeMethod);
		}		
	}
	
	else
	{
		int numVertsInCache = hkWiiLockedCacheNonInterleaved::computeNumVertsInCache(parameters);
		if (numVertsInCache<MIN_NUM_CACHED_VERTS)
		{
			HK_WARN_ONCE (0x79d9603c, "Cannot fit non-interleaved input buffer with P stride " << positionsInStride << \
			 			" bytes and N stride " << normalsInStride <<  " bytes in the 16Kb locked cache, therefore reverting to generic skinning for this buffer" );
			return false;
		}
		
		hkWiiLockedCacheNonInterleaved lockedCache(numVertsInCache, parameters);
			
		DCFlushRange((u8*)normalsIn, numVertices * normalsInStride);
			
		if (writeGatherEnabled)
		{	
			hkWiiWriteToPipe writeMethod;
			
			hkSkinWii_PN(parameters, lockedCache, writeMethod);
		}
		else
		{	
			hkWiiWriteToMemory writeMethod;
			
			hkSkinWii_PN(parameters, lockedCache, writeMethod);
		}
	}
	
	return true;
}


static bool hkWiiSkinning_P(const hkSkinOperator::Parameters& parameters)
{
	const hkUint32 numVertices = parameters.m_numVertices;

	const hkSkinOperator::SkinStream& inputStream = parameters.m_input;
	const void* positionsIn = hkAddByteOffsetConst(inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_start, inputStream.m_positions.m_offset);
	const hkUint32 positionsInStride = inputStream.m_buffers[inputStream.m_positions.m_bufferIndex].m_stride;
	
	// We assume skinning via the locked cache is not likely to be fast if less than MIN_NUM_CACHED_VERTS fit in the cache.
	const int MIN_NUM_CACHED_VERTS = 16;
	
	int numVertsInCache = hkWiiLockedCacheInterleaved::computeNumVertsInCache(parameters);
	if (numVertsInCache<MIN_NUM_CACHED_VERTS)
	{	
		HK_WARN_ONCE (0x6bd6a8da, "Cannot fit input buffer with stride " << positionsInStride << \
			 			" bytes in the 16Kb locked cache, therefore reverting to generic skinning for this buffer" );
		return false;
	}
		
	hkWiiLockedCacheInterleaved lockedCache(numVertsInCache, parameters);	
	
	// Flush all locations we will (potentially) need to read from the cache, to make sure they are read fresh from main memory.
	DCFlushRange((u8*)parameters.m_compositeMatrices,        parameters.m_numBones * sizeof(hkMatrix4));
	DCFlushRange((u8*)parameters.m_boneInfluences,           numVertices * 4 * sizeof(hkSkinOperator::BoneInfluence));
	DCFlushRange((u8*)parameters.m_startInfluencePerVertex , numVertices * sizeof(hkUint16));
	DCFlushRange((u8*)positionsIn,                           numVertices * positionsInStride);	
	
	hkWiiWriteToMemory writeMethod;
	
	hkSkinWii_P(parameters, lockedCache, writeMethod);
	
	return true;
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
