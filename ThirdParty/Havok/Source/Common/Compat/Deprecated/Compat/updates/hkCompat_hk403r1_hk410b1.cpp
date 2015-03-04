/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Compat/Deprecated/Compat/hkCompatUtil.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
	
#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

namespace hkHavok403r1Classes
{
	extern hkClass hkConvexShapeClass;
}

namespace hkCompat_hk403r1_hk410b1
{
	class CompatVersionContext
	{
	public:

		CompatVersionContext( hkArray<hkVariant>& objectsInOut )
		{
			for (int i = 0; i < objectsInOut.getSize(); ++i )
			{
				m_classFromObject.insert(objectsInOut[i].m_object, objectsInOut[i].m_class);
			}
		}

		const hkClass* findClassFromOldObject( const void* oldObject )
		{
			return m_classFromObject.getWithDefault(oldObject, HK_NULL);
		}

		static hkBool classIsDerivedFrom(const hkClass* klass, const char* baseName)
		{
			while (klass && hkString::strCmp(klass->getName(), baseName) != 0)
			{
				klass = klass->getParent();
			}
			return klass != HK_NULL;
		}

		hkBool oldObjectIsA(const void* oldObjectToCheck, const char* typeName)
		{
			if( const hkClass* c = findClassFromOldObject(oldObjectToCheck) )
			{
				return classIsDerivedFrom(c, typeName);
			}
			return false;
		}

		hkPointerMap<const void*, const hkClass*> m_classFromObject;
	};

	static CompatVersionContext* s_compatVersionContext;

struct DummyQuantizeDesc
{
	hkUint8  bitWidth;	// The number of bits for each quantized element
	hkUint8  preserved;	// The number of unquantized reals at the start of the stream
	hkReal   scale;		// The scale of the data = (max-min)
	hkReal   offset;	// The offset of the data (zero is preserved)
};

hkUint32 calcQuantizedSize(hkUint32 nValues, const DummyQuantizeDesc& desc)
{
	HK_ASSERT2 (0x6fe654e5, (desc.bitWidth > 0) && (desc.bitWidth < 17), "Only 1 to 16 bit quantization supported");

	// begin with space required for preserve
	int spaceRequired = desc.preserved * sizeof(hkReal);

	// calculate total bit space required
	int bitSpaceRequired = desc.bitWidth * ( nValues - desc.preserved );
	int byteSpaceRequired = ( bitSpaceRequired + 7 ) / 8;

	// total space required
	spaceRequired += byteSpaceRequired;

	return spaceRequired;
}

static hkInt32 getTypeValueFromEnumName(const hkClassAccessor& memAcc, const char* typeName)
{
	int ret;
	HK_ON_DEBUG(hkResult res = )memAcc.member("type").getClassMember().getEnumClass().getValueOfName(typeName, &ret);
	HK_ASSERT(0x1b92e41e, res == HK_SUCCESS);
	return hkInt32(ret);
}

template <class T>
inline const hkArray<T>& getArray(const hkClassMemberAccessor& memAcc)
{
	return *static_cast<hkArray<T>*>(memAcc.asRaw());
}

static HK_FORCE_INLINE void byteswap(unsigned char& a, unsigned char& b)
{
	unsigned char t = a;
	a = b;
	b = t;
}

template <class T>
inline void endianSwap(unsigned char* bytes, int numBytes)
{
#if HK_ENDIAN_BIG
	switch (sizeof(T))
	{
	case 1:
		break;
	case 2:
		for (int i=0; i < numBytes; i+=2)
		{
			byteswap(bytes[i],bytes[i+1]);
		}
		break;
	case 4:
		for (int i=0; i < numBytes; i+=4)
		{
			byteswap(bytes[i],bytes[i+3]);
			byteswap(bytes[i+1],bytes[i+2]);
		}
		break;
	default:
		HK_ERROR(0x45345432, "Can't handle endian");
	}
#endif
}

template <class T>
inline hkInt32 copyArrayDataToNewClass(
	const hkClassMemberAccessor& oldMemAcc, hkInt32 sizeInBytes,
	hkArray<hkUint8>& destBuffer, hkInt32 bufferOffset,
	const hkClassMemberAccessor& newMemIdxAcc)
{
	const hkArray<T>& srcArray = getArray<T>(oldMemAcc);
	hkInt32 alignedSize = HK_NEXT_MULTIPLE_OF(4, sizeInBytes);
	destBuffer.expandBy(alignedSize);
	hkString::memCpy( destBuffer.begin()+bufferOffset, srcArray.begin(), sizeInBytes );
	endianSwap<T>( destBuffer.begin()+bufferOffset, sizeInBytes);

	newMemIdxAcc.asInt32() = bufferOffset;
	return alignedSize;
}

template <class T>
inline hkInt32 copyArrayDataToNewClass(
	const hkClassMemberAccessor& oldMemAcc,
	hkArray<hkUint8>& destBuffer, hkInt32 bufferOffset,
	const hkClassMemberAccessor& newMemIdxAcc)
{
	const hkArray<T>& srcArray = getArray<T>(oldMemAcc);
	hkInt32 sizeInBytes = srcArray.getSize() * sizeof(T);
	hkInt32 alignedSize = HK_NEXT_MULTIPLE_OF(4, sizeInBytes);
	destBuffer.expandBy(alignedSize);
	hkString::memCpy( destBuffer.begin()+bufferOffset, srcArray.begin(), sizeInBytes );
	endianSwap<T>( destBuffer.begin()+bufferOffset, sizeInBytes);

	newMemIdxAcc.asInt32() = bufferOffset;
	return alignedSize;
}

inline hkInt32 copyBitWidthArrayDataToNewClass(
	const hkClassMemberAccessor& oldMemAcc, hkInt32 numOfItems, hkUint8 maxBitWidth,
	hkArray<hkUint8>& destBuffer, hkInt32 bufferOffset,
	const hkClassMemberAccessor& newMemIdxAcc)
{
	const hkArray<hkUint8>& srcArray = getArray<hkUint8>(oldMemAcc);
	hkInt32 sizeInBytes = numOfItems * sizeof(hkUint8);
	hkInt32 alignedSize = HK_NEXT_MULTIPLE_OF(4, sizeInBytes);
	destBuffer.expandBy(alignedSize);
	
	// Check if we have all of the bitwidths (this won't be true for old assets)
	if( srcArray.getSize() == numOfItems )
	{
		hkString::memCpy( destBuffer.begin()+bufferOffset, srcArray.begin(), sizeInBytes );
	}
	else
	{
		hkString::memSet( destBuffer.begin()+bufferOffset, maxBitWidth, sizeInBytes );
	}
	newMemIdxAcc.asInt32() = bufferOffset;
	return alignedSize;
}

static void SkeletalAnimation_403r1_410b1(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& )
{

	hkClassAccessor newInstance(newObj);
	hkInt32& type = newInstance.member("type").asInt32();

	if( hkString::strCmp(newObj.m_class->getName(),"hkInterleavedSkeletalAnimation") == 0 )
	{
		type = hkCompat_hk403r1_hk410b1::getTypeValueFromEnumName( newObj, "HK_INTERLEAVED_ANIMATION" );
	}
	else if( hkString::strCmp(newObj.m_class->getName(),"hkDeltaCompressedSkeletalAnimation") == 0 )
	{
		type = hkCompat_hk403r1_hk410b1::getTypeValueFromEnumName( newObj, "HK_DELTA_COMPRESSED_ANIMATION" );
	}
	else if( hkString::strCmp(newObj.m_class->getName(),"hkWaveletSkeletalAnimation") == 0 )
	{
		type = hkCompat_hk403r1_hk410b1::getTypeValueFromEnumName( newObj, "HK_WAVELET_COMPRESSED_ANIMATION" );
	}
	else
	{
		type = hkCompat_hk403r1_hk410b1::getTypeValueFromEnumName( newObj, "HK_UNKNOWN_ANIMATION" );
	}
}

static void DeltaAnimation_403r1_410b1(hkVariant& oldObj, hkVariant& newObj, hkObjectUpdateTracker& tracker )
{
	hkClassAccessor oldInstance(oldObj);
	hkClassAccessor newInstance(newObj);

	SkeletalAnimation_403r1_410b1(oldObj, newObj, tracker);

	hkClassAccessor oldQFormat = oldInstance.member("qFormat").object();
	hkClassAccessor newQFormat = newInstance.member("qFormat").object();

	hkClassMemberAccessor oldOffsetArray = oldQFormat.member("offset");
	const hkArray<hkReal>& offsetArray = getArray<hkReal>(oldOffsetArray);
	hkUint32 numD = offsetArray.getSize();
	newQFormat.member("numD").asInt32() = numD;

	hkArray<hkUint8> dataBuffer;
	hkInt32 offset = 0;

	// other members
	//
	// copy the original's offset, scale, and bitWidth arrays
	offset += copyArrayDataToNewClass<hkReal>(oldQFormat.member("offset"), numD * sizeof(hkReal),
												dataBuffer, offset, newQFormat.member("offsetIdx"));
	offset += copyArrayDataToNewClass<hkReal>(oldQFormat.member("scale"), numD * sizeof(hkReal),
												dataBuffer, offset, newQFormat.member("scaleIdx"));

	// copy the static DOFs
	offset += copyArrayDataToNewClass<hkReal>(oldInstance.member("staticDOFs"),
												dataBuffer, offset, newInstance.member("staticDOFsIdx"));

	// copy the static mask
	offset += copyArrayDataToNewClass<hkUint16>(oldInstance.member("staticMask"),
		dataBuffer, offset, newInstance.member("staticMaskIdx"));


	offset += copyBitWidthArrayDataToNewClass(oldQFormat.member("bitWidth"), numD, newQFormat.member("maxBitWidth").asUint8(),
		dataBuffer, offset, newQFormat.member("bitWidthIdx"));

	// Copy the quantized data (this goes last since we won't DMA all of it each frame)
	offset += copyArrayDataToNewClass<hkUint8>(oldInstance.member("quantizedData"),
												dataBuffer, offset, newInstance.member("quantizedDataIdx"));

	// Compute m_lastBlockSize
	hkUint32 lastBlockSize = 0;
	hkInt32 lastBlockFrames = newInstance.member("numberOfPoses").asInt32() % newInstance.member("blockSize").asInt32();
	DummyQuantizeDesc desc;
	desc.preserved = newQFormat.member("preserved").asUint8();
	hkUint8* bitWidthPtr = dataBuffer.begin() + hkUint32(newQFormat.member("bitWidthIdx").asInt32());
	for (unsigned int i=0; i<numD; i++)
	{
		desc.bitWidth = bitWidthPtr[i];
		lastBlockSize += calcQuantizedSize( lastBlockFrames, desc );
	}
	newInstance.member("lastBlockSize").asInt32() = lastBlockSize;

	// finally, assign the data buffer to the new object's new class member
	hkClassMemberAccessor::SimpleArray& newDataBuffer = newInstance.member("dataBuffer").asSimpleArray();
	newDataBuffer.size = dataBuffer.getSize();
	newDataBuffer.data = hkAllocateChunk<hkUint8>( newDataBuffer.size, HK_MEMORY_CLASS_ANIM_COMPRESSED );
	hkString::memCpy(newDataBuffer.data, dataBuffer.begin(), newDataBuffer.size);
	// keep tracking of the data buffer pointer
	tracker.objectPointedBy( newDataBuffer.data, newInstance.member("dataBuffer").getAddress() );
	tracker.addChunk( newDataBuffer.data,  newDataBuffer.size, HK_MEMORY_CLASS_ANIM_COMPRESSED );
}

static void WaveletAnimation_403r1_410b1(
		hkVariant& oldObj,
		hkVariant& newObj,
		hkObjectUpdateTracker& tracker )
{
	hkClassAccessor newInstance(newObj);
	hkClassAccessor oldInstance(oldObj);

	SkeletalAnimation_403r1_410b1(oldObj, newObj, tracker);
	// qFormat
	hkClassAccessor oldQFormat = oldInstance.member("qFormat").object();
	hkClassAccessor newQFormat = newInstance.member("qFormat").object();

	hkClassMemberAccessor oldOffsetArray = oldQFormat.member("offset");
	const hkArray<hkReal>& offsetArray = getArray<hkReal>(oldOffsetArray);
	hkUint32 numD = offsetArray.getSize();
	newQFormat.member("numD").asInt32() = numD;

	hkArray<hkUint8> dataBuffer;
	hkInt32 offset = 0;

	// other members
	//
	// copy the original's offset, scale
	offset += copyArrayDataToNewClass<hkReal>(oldQFormat.member("offset"), numD * sizeof(hkReal),
												dataBuffer, offset, newQFormat.member("offsetIdx"));
	offset += copyArrayDataToNewClass<hkReal>(oldQFormat.member("scale"), numD * sizeof(hkReal),
												dataBuffer, offset, newQFormat.member("scaleIdx"));
	// copy the static DOFs
	offset += copyArrayDataToNewClass<hkReal>(oldInstance.member("staticDOFs"),
												dataBuffer, offset, newInstance.member("staticDOFsIdx"));

	hkInt32 bytesCopied;

	// copy the quantized data offsets
	bytesCopied = copyArrayDataToNewClass<hkUint32>(oldInstance.member("blockIndex"),
												dataBuffer, offset, newInstance.member("blockIndexIdx"));
	newInstance.member("blockIndexSize").asInt32() = bytesCopied / sizeof(hkUint32);
	offset += bytesCopied;

	// copy the static mask
	offset += copyArrayDataToNewClass<hkUint16>(oldInstance.member("staticMask"),
		dataBuffer, offset, newInstance.member("staticMaskIdx"));

	// Copy the bit width array
	offset += copyBitWidthArrayDataToNewClass(oldQFormat.member("bitWidth"), numD, newQFormat.member("maxBitWidth").asUint8(),
		dataBuffer, offset, newQFormat.member("bitWidthIdx"));

	// copy the quantized data (this goes last since we won't DMA all of it each frame)
	bytesCopied = copyArrayDataToNewClass<hkUint8>(oldInstance.member("quantizedData"),
												dataBuffer, offset, newInstance.member("quantizedDataIdx"));
	newInstance.member("quantizedDataSize").asInt32() = bytesCopied / sizeof(hkUint8);
	offset += bytesCopied;

	// finally, assign the data buffer to the new object's new class member
	hkClassMemberAccessor::SimpleArray& newDataBuffer = newInstance.member("dataBuffer").asSimpleArray();
	newDataBuffer.size = dataBuffer.getSize();
	newDataBuffer.data = hkAllocateChunk<hkUint8>( newDataBuffer.size, HK_MEMORY_CLASS_ANIM_COMPRESSED );
	hkString::memCpy(newDataBuffer.data, dataBuffer.begin(), newDataBuffer.size);
	// keep tracking of the data buffer pointer
	tracker.objectPointedBy( newDataBuffer.data, newInstance.member("dataBuffer").getAddress() );
	tracker.addChunk( newDataBuffer.data,  newDataBuffer.size, HK_MEMORY_CLASS_ANIM_COMPRESSED );
}

#define REMOVED(TYPE) { 0,0, hkVersionRegistry::VERSION_REMOVED, TYPE, HK_NULL }
#define BINARY_IDENTICAL(OLDSIG,NEWSIG,TYPE) { OLDSIG, NEWSIG, hkVersionRegistry::VERSION_MANUAL, TYPE, HK_NULL }


static void hkShape_403r1_410b1(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& )
{
	HK_COMPILE_TIME_ASSERT(sizeof(void*) == sizeof(hkUlong));
	// do nothing, the hkShape::m_userData has the same size (sizeof(void*) == sizeof(hkUlong))
}

struct OldArrayType
{
	void* shape;
	hkUint32* unusedFilterInfo;
};

static void hkConvexListShape_403r1_410b1(
	hkVariant& oldObj,
	hkVariant& newObj,
	hkObjectUpdateTracker& )
{

	hkClassMemberAccessor newChildShapes(newObj, "childShapes");
	hkClassMemberAccessor oldChildShapes(oldObj, "childInfo");
	if( newChildShapes.isOk() && oldChildShapes.isOk() )
	{

		hkArray<OldArrayType>& oldArray = *(hkArray<OldArrayType>*)oldChildShapes.asRaw();
		hkArray<void*>& newArray = *(new (newChildShapes.asRaw()) hkArray<void*>());

		newArray.setSize( oldArray.getSize());
		for (int i = 0; i < oldArray.getSize(); ++i )
		{
			newArray[i] = oldArray[i].shape;
		}

		hkVariant v;
		v.m_class = &hkHavok403r1Classes::hkConvexShapeClass;
		v.m_object = oldArray[0].shape;
		hkClassMemberAccessor newRadius(newObj, "radius");
		hkClassMemberAccessor oldRadius(v, "radius");
		newRadius.asReal() = oldRadius.asReal();
	}
}

static void hkEntity_403r1_410b1(
								 hkVariant& oldObj,
								 hkVariant& newObj,
								 hkObjectUpdateTracker& )
{
	hkClassAccessor newInstance(newObj);
	hkClassAccessor oldInstance(oldObj);

	// get the type of the deactivator. (which is an object)
	// if the type != spatial than set m_motion.m_deactivationIntegrateCounter to 0xff
	//
	const void* oldDeactivatorObjPtr = oldInstance.member("deactivator").asPointer();
	if( s_compatVersionContext && s_compatVersionContext->oldObjectIsA(oldDeactivatorObjPtr, "hkSpatialRigidBodyDeactivator") == false )
	{
		hkClassMemberAccessor newMotionInstance = newInstance.member("motion");
		void* newMotionObjPtr = newMotionInstance.getAddress();
		const hkClass& newMotionKlass = newMotionInstance.getClassMember().getStructClass();
		hkClassAccessor	newMotion(newMotionObjPtr, &newMotionKlass);
		newMotion.member("deactivationIntegrateCounter").asUint8() = 0xff;
	}
}

static const hkVersionRegistry::ClassAction s_updateActions[] =
{
	// common
	{ 0x8bdd3e9a, 0x8bdd3e9a, hkVersionRegistry::VERSION_VARIANT, "hkBoneAttachment", HK_NULL },
	{ 0xf598a34e, 0xf598a34e, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainer", HK_NULL },
	{ 0x853a899c, 0x853a899c, hkVersionRegistry::VERSION_VARIANT, "hkRootLevelContainerNamedVariant", HK_NULL }, 
	{ 0x3d43489c, 0x3d43489c, hkVersionRegistry::VERSION_VARIANT, "hkxMaterial", HK_NULL }, 
	{ 0x914da6c1, 0x914da6c1, hkVersionRegistry::VERSION_VARIANT, "hkxAttribute", HK_NULL },
	{ 0x1667c01c, 0x1667c01c, hkVersionRegistry::VERSION_VARIANT, "hkxAttributeGroup", HK_NULL },
	{ 0x0a62c79f, 0x0a62c79f, hkVersionRegistry::VERSION_VARIANT, "hkxNode", HK_NULL }, 
	{ 0xe085ba9f, 0xe085ba9f, hkVersionRegistry::VERSION_VARIANT, "hkxMaterialTextureStage", HK_NULL },
	{ 0x57061454, 0x57061454, hkVersionRegistry::VERSION_HOMOGENEOUSARRAY, "hkxVertexBuffer", HK_NULL },
	{ 0x91f785ca, 0x629b527c, hkVersionRegistry::VERSION_COPY, "hkConvexListShape", hkConvexListShape_403r1_410b1 },
	{ 0x804c9b06, 0xfdcdf165, hkVersionRegistry::VERSION_COPY, "hkWorldCinfo", HK_NULL },

	{ 0x01ed92da, 0x008d5bf4, hkVersionRegistry::VERSION_COPY, "hkEntity", hkEntity_403r1_410b1 },

	{ 0x332f16fa, 0xc9c72e9e, hkVersionRegistry::VERSION_COPY, "hkMotionState", HK_NULL },
	{ 0x179f1a0b, 0x66989e6a, hkVersionRegistry::VERSION_COPY, "hkMotion", HK_NULL },


	// hkbase
	BINARY_IDENTICAL( 0xa52796eb, 0xe1ac568b, "hkClass" ),	// TYPE_ULONG
	BINARY_IDENTICAL( 0x2e50284b, 0xd2665ef8, "hkClassMember" ), // TYPE_ULONG

	// hkdynamics
	BINARY_IDENTICAL( 0xd7d12547, 0x80372d43, "hkBridgeConstraintAtom" ), // fixed +nosave
	{ 0x93b56af0, 0x4136ee13, hkVersionRegistry::VERSION_COPY, "hkShapePhantom", HK_NULL }, // hkMotionState changed
	BINARY_IDENTICAL( 0xba9333e4, 0x1762d81f, "hkBridgeAtoms" ), // hkBridgeConstraintAtom fixed +nosave
	BINARY_IDENTICAL( 0x4fca7e0a, 0x94030c62, "hkStiffSpringChainData" ), // hkBridgeConstraintAtom fixed +nosave
	BINARY_IDENTICAL( 0xd0ffea9e, 0x5bde2861, "hkPoweredChainData" ), // hkBridgeConstraintAtom fixed +nosave
	BINARY_IDENTICAL( 0x8907e64c, 0xca45a993, "hkPointToPathConstraintData" ), // hkBridgeConstraintAtom fixed +nosave
	BINARY_IDENTICAL( 0xa80474ce, 0xe26600ea, "hkMalleableConstraintData" ), // hkBridgeConstraintAtom fixed +nosave
	BINARY_IDENTICAL( 0x1abb6f60, 0xc781e207, "hkGenericConstraintData" ), // hkBridgeConstraintAtom fixed +nosave
	BINARY_IDENTICAL( 0x816f4533, 0xbf49e9bd, "hkBreakableConstraintData" ), // hkBridgeConstraintAtom fixed +nosave
	BINARY_IDENTICAL( 0x6286c3ed, 0xc058c764, "hkBallSocketChainData" ), // hkBridgeConstraintAtom fixed +nosave

	// hkcollide
	//BINARY_IDENTICAL( 0x9ab27645, 0x99dde7b9, "hkShape" ), // m_userData (void* -> hkUlong)
	{ 0x9ab27645, 0x99dde7b9, hkVersionRegistry::VERSION_MANUAL, "hkShape", hkShape_403r1_410b1 }, // m_userData (void* -> hkUlong)

	
	{ 0xfdee6f42, 0xde5deced, hkVersionRegistry::VERSION_COPY, "hkDeltaCompressedSkeletalAnimation", DeltaAnimation_403r1_410b1 },	
	{ 0x309d43f9, 0x724a7561, hkVersionRegistry::VERSION_COPY, "hkDeltaCompressedSkeletalAnimationQuantizationFormat", HK_NULL },
	{ 0xfd1fe320, 0xacee09b8, hkVersionRegistry::VERSION_COPY, "hkWaveletSkeletalAnimation", WaveletAnimation_403r1_410b1 },		
	{ 0x309d43f9, 0x724a7561, hkVersionRegistry::VERSION_COPY, "hkWaveletSkeletalAnimationQuantizationFormat", HK_NULL },
	{ 0xb1aac849, 0x24b3d6bc, hkVersionRegistry::VERSION_COPY, "hkSkeletalAnimation", SkeletalAnimation_403r1_410b1 },

	{ 0, 0, 0, HK_NULL, HK_NULL }
};

static const hkVersionRegistry::ClassRename s_renames[] =
{
	{ HK_NULL, HK_NULL }
};

#define HK_COMPAT_VERSION_FROM hkHavok403r1Classes
#define HK_COMPAT_VERSION_TO hkHavok410b1Classes
#define HK_COMPAT_OPTIONAL_UPDATE_FUNC update

extern hkVersionRegistry::UpdateDescription hkVersionUpdateDescription;

static hkResult HK_CALL update(
	hkArray<hkVariant>& objectsInOut,
	hkObjectUpdateTracker& tracker )
{
	CompatVersionContext context(objectsInOut);
	s_compatVersionContext = &context;
	hkResult res = hkVersionUtil::updateSingleVersion( objectsInOut, tracker, hkVersionUpdateDescription, hkVersionRegistry::getInstance().getClassNameRegistry(HK_COMPAT_VERSION_TO::VersionString) );
	s_compatVersionContext = HK_NULL;

	return res;
}

#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO
} // namespace hkCompat_hk403r1_hk410b1

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
