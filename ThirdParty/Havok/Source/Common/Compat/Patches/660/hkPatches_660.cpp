/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/KeyCode.h>

// Registration function is at the end of the file

static void hkxSparselyAnimatedString_0_to_1(hkDataObject& obj)
{
	hkDataArray oldStrings = obj["old_strings"].asArray();
	hkDataArray newStrings = obj["strings"].asArray();
	newStrings.setSize(oldStrings.getSize());
	for( int i = 0; i < oldStrings.getSize(); ++i )
	{
		newStrings[i] = oldStrings[i].asObject()["string"].asString();
	}
}

namespace hkxVertexBuffer_0_to_1_Util
{
	enum DataType
	{
		HKX_DT_NONE = 0,
		HKX_DT_UINT8, // only used for four contiguous hkUint8s, hkUint8[4]
		HKX_DT_INT16, // only used for old style quantized tcoords (0x7fff maps to 10.0f), so div by 3276.7f to get the float tcoords. Deprecated.
		HKX_DT_UINT32,
		HKX_DT_FLOAT,
		HKX_DT_FLOAT2, // for tex coords 
		HKX_DT_FLOAT3, // will always be 16byte aligned, so you can treat as a hkVector4 (with undefined w, with SIMD enabled etc)
		HKX_DT_FLOAT4  // will always be 16byte aligned, so you can treat as a hkVector4 (with SIMD enabled etc)
	};
	static const int s_byteStrideFromDataType[] =
	{
		-1,
		4,
		4,
		4,
		4,
		8,
		16,
		16
	};

	enum DataUsage
	{
		HKX_DU_NONE = 0,
		HKX_DU_POSITION = 1,
		HKX_DU_COLOR = 2,    // first color always can be assumed to be per vertex Diffuse, then per vertex Specular (rare)
		HKX_DU_NORMAL = 4,
		HKX_DU_TANGENT = 8,
		HKX_DU_BINORMAL = 16, // aka BITANGENT
		HKX_DU_TEXCOORD = 32, // in order, 0,1,2, etc of the texture channels. Assumed to be 2D, [u,v], in most cases
		HKX_DU_BLENDWEIGHTS = 64,  // usually 4 weights, but 3 can be stored with 1 implied. Can be stored as 4*uint8, so quantized where 1.0f => 0xff (255),
		HKX_DU_BLENDINDICES = 128, // usually 4 hkUint8s in a row. So can reference 256 blend transforms (bones)
		HKX_DU_USERDATA = 256
	};
}

static void hkxVertexBuffer_0_to_1(hkDataObject& obj)
{
	const hkDataWorld* world = obj.getClass().getWorld();

	hkDataArray oldData = obj["vertexData"].asArray();
	hkDataClass oldClass = oldData.getClass();

	hkDataClass vertexDataClass( world->findClass("hkxVertexBufferVertexData") );
	hkDataObject newData = world->newObject( vertexDataClass );
	newData["numVerts"] = oldData.getSize();
	obj["data"] = newData;

	hkDataObject oldDesc = obj["vertexDesc"].asObject();
	hkDataArray oldDecls = oldDesc["decls"].asArray();

	hkDataClass descriptionClass( world->findClass("hkxVertexDescription") );
	hkDataObject newDesc = world->newObject( descriptionClass );
	obj["desc"] = newDesc;

	hkDataArray newDecls = newDesc["decls"].asArray();
	newDecls.setSize( oldDecls.getSize() );
	int numVertices = oldData.getSize();
	int curTextureChannel = 0;
	int curColorChannel = 0;

	for( int declIndex = 0; declIndex < oldDecls.getSize(); ++declIndex )
	{
		hkDataObject oldDecl = oldDecls[declIndex].asObject();
		hkDataObject newDecl = newDecls[declIndex].asObject();
		newDecl["type"] = oldDecl["type"];
		newDecl["usage"] = oldDecl["usage"];
		newDecl["byteStride"] = hkxVertexBuffer_0_to_1_Util::s_byteStrideFromDataType[ oldDecl["type"].asInt() ];
		int dataUsage = oldDecl["usage"].asInt();
		switch( dataUsage )
		{
		case hkxVertexBuffer_0_to_1_Util::HKX_DU_POSITION:
		case hkxVertexBuffer_0_to_1_Util::HKX_DU_NORMAL:
		case hkxVertexBuffer_0_to_1_Util::HKX_DU_BINORMAL:
		case hkxVertexBuffer_0_to_1_Util::HKX_DU_TANGENT:
			{
				const char* name =
					dataUsage == hkxVertexBuffer_0_to_1_Util::HKX_DU_POSITION ? "position"
					: dataUsage == hkxVertexBuffer_0_to_1_Util::HKX_DU_NORMAL ? "normal"
					: dataUsage == hkxVertexBuffer_0_to_1_Util::HKX_DU_BINORMAL ? "binormal"
					: dataUsage == hkxVertexBuffer_0_to_1_Util::HKX_DU_TANGENT ? "tangent"
					: HK_NULL;
				hkDataArray o = oldData.swizzleObjectMember( name );
				hkDataArray n = newData["vectorData"].asArray();
				int start = n.getSize();
				newDecl["byteOffset"] = n.getSize() * hkSizeOf(hkVector4);
				n.setSize( n.getSize() + numVertices );
				for( int i = 0; i < numVertices; ++i )
				{
					n[i+start] = o[i].asVector4();
				}
				break;
			}
		case hkxVertexBuffer_0_to_1_Util::HKX_DU_COLOR:
			{
				char name[] = {'d', 'i', 'f', 'f', 'u', 's', 'e', char('A'+curColorChannel), 0 };
				if( curTextureChannel == 0 && oldClass.getMemberIndexByName(name) == -1 )
				{
					HK_ASSERT(0x2e7c90d3, name[7] == 'A');
					name[7] = 0; // just "diffuse" not "diffuseA"
				}
				hkDataArray o = oldData.swizzleObjectMember( name );
				hkDataArray n = newData["uint32Data"].asArray();
				int start = n.getSize();
				newDecl["byteOffset"] = n.getSize() * hkSizeOf(hkUint32);
				n.setSize( n.getSize() + numVertices );
				for( int i = 0; i < numVertices; ++i )
				{
					n[i+start] = o[i].asInt();
				}
				curColorChannel += 1;
				break;
			}
		case hkxVertexBuffer_0_to_1_Util::HKX_DU_TEXCOORD:
			{
				char uname[3] = {'u', char('0'+curTextureChannel) };
				char vname[3] = {'v', char('0'+curTextureChannel) };
				if( curTextureChannel == 0 && oldClass.getMemberIndexByName(uname) == -1 )
				{
					uname[1] = 0; // just "u" not "u0"
					vname[1] = 0;
				}
				hkDataArray ud = oldData.swizzleObjectMember(uname);
				hkDataArray vd = oldData.swizzleObjectMember(vname);
				hkDataArray n = newData["floatData"].asArray();
				int start = n.getSize();
				newDecl["byteOffset"] = n.getSize() * hkSizeOf(hkReal);
				n.setSize( n.getSize() + numVertices*4 );
				for( int i = 0; i < numVertices; ++i )
				{
					n[i*2+start  ] = ud[i].asReal();
					n[i*2+start+1] = vd[i].asReal();
				}
				curTextureChannel += 1;
				break;
			}
		case hkxVertexBuffer_0_to_1_Util::HKX_DU_BLENDWEIGHTS:
			{
				hkDataArray w0 = oldData.swizzleObjectMember("w0");
				hkDataArray w1 = oldData.swizzleObjectMember("w1");
				hkDataArray w2 = oldData.swizzleObjectMember("w2");
				hkDataArray w3 = oldData.swizzleObjectMember("w3");

				hkDataArray n = newData["uint8Data"].asArray();
				int start = n.getSize();
				newDecl["byteOffset"] = n.getSize() * 1;
				n.setSize( n.getSize() + numVertices*4 );
				for( int i = 0; i < numVertices; ++i )
				{
					n[i*4+start  ] = w0[i].asInt();
					n[i*4+start+1] = w1[i].asInt();
					n[i*4+start+2] = w2[i].asInt();
					n[i*4+start+3] = w3[i].asInt();
				}
				break;
			}
		case hkxVertexBuffer_0_to_1_Util::HKX_DU_BLENDINDICES:
			{
				hkDataArray i0 = oldData.swizzleObjectMember("i0");
				hkDataArray i1 = oldData.swizzleObjectMember("i1");
				hkDataArray i2 = oldData.swizzleObjectMember("i2");
				hkDataArray i3 = oldData.swizzleObjectMember("i3");

				hkDataArray n = newData["uint8Data"].asArray();
				int start = n.getSize();
				newDecl["byteOffset"] = n.getSize() * 1;
				n.setSize( n.getSize() + numVertices*4 );
				for( int i = 0; i < numVertices; ++i )
				{
					n[i*4+start  ] = i0[i].asInt();
					n[i*4+start+1] = i1[i].asInt();
					n[i*4+start+2] = i2[i].asInt();
					n[i*4+start+3] = i3[i].asInt();
				}
				break;
			}
		default:
			{
				HK_ASSERT(0x5bca7458, 0);
			}
		}
	}
}

void HK_CALL registerCommonPatches_660(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/660/hkPatches_660.cxx>
#	include <Common/Serialize/Version/hkVersionPatchManager.cxx>
#	undef HK_PATCHES_FILE
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
