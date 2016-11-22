import "regent"

-- Helper modules to handle PNG files and command line arguments
local png        = require("png_util")
local WaveletConfig = require("wavelet_config")

-- Some C APIs
local c     = regentlib.c
local sqrt  = regentlib.sqrt(double)
local cmath = terralib.includec("math.h")
local PI = cmath.M_PI
local unistd = terralib.includec("unistd.h")


-- Field space for pixels
-- Should be the only required field because we make changes in place.
fspace Pixel
{
  value      : uint8;    -- value pixel in 8-bit gray scale
}

-- FIXME: Which parameters do we need to provide in comparison to gfxc?
--struct Parameters
--{
  --step : int32;
  --size : int2d;
--}

task initialize(r_image : region(ispace(int2d), Pixel),
                filename : rawstring)
where
  reads writes(r_image)
do
  png.read_png_file(filename,
                    __physical(r_image.value),
                    __fields(r_image.value),
                    r_image.bounds)
end

task liftX(r_image    : region(ispace(int2d), Pixel), 
          step : int32)
where
  reads writes(r_image.value)
do 
  var ts_start = c.legion_get_current_time_in_micros()
  c.printf("starting liftX!\n")
  
  var size_x : int32 = r_image.bounds.hi.x
  var size_y : int32 = r_image.bounds.hi.y

  for y = 0, size_y, step do
    
    var base = 0
    var base1 = base - step
    var base2 = base + step
    -- odd loop? Only when step is 1 I guess.
    
    var x : int32
    for x = step, size_x - step, step*2 do
      r_image[{base+x, y}].value -= (r_image[{base1+x, y}].value + r_image[{base2+x, y}].value)/2
    end
    
    -- Edge condition.
    if (x < size_x) then
      r_image[{base+x, y}].value -= r_image[{base1+x, y}].value 
    end
    
    -- even loop?
    for x = step*2, size_x - step, step*2 do
      r_image[{base+x, y}].value += (r_image[{base1+x, y}].value + r_image[{base2+x, y}].value)/4 
    end
    
    if (x < size_x) then
      r_image[{base+x, y}].value += r_image[{base1+x, y}].value / 2 
    end
    -- Edge condition.
    
  end

  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Lift x took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
end

-- Vertical lifting. Slightly more complicated than Horizontal lifting.
--
task liftY(r_image    : region(ispace(int2d), Pixel), 
          step : int32)
where
  reads writes(r_image.value)
do 

  var ts_start = c.legion_get_current_time_in_micros()
  c.printf("starting liftY!\n")
  
  var y0 : int32 = r_image.bounds.lo.y
  var x0 : int32 = r_image.bounds.lo.x
  -- I think this should be always true?
  --assert (y0 == 0 && x0 == 0)

  var size_x : int32 = r_image.bounds.hi.x
  var size_y : int32 = r_image.bounds.hi.y

  for y = step, size_y, step*2 do
        
    var base :int2d = {x0, y0 + y}
    -- FIXME: Does this always stay within range?
    var c1base :int2d = {x0, y0 + y - step}
    var c2base :int2d = c1base
    
    if (y + step) < size_y then 
      c2base = {x0, y0 + y + step}
    end
      
    var x : int32

    -- Linear filtering
    -- If we did only a few columns at a time? 
    for x = 0, size_x, step do
      r_image[{base.x + x, base.y}].value -= (r_image[c1base + {x,0}].value + r_image[c2base + {x,0}].value) / 2
    end 
  end
  
  -- Do we really need a separate loop for this?
  for y = step*2, size_y, step*2 do
    var base :int2d = {x0 , y0 + y}
    -- FIXME: Does this always stay within range?
    var g1base :int2d = {x0, y0 + y - step}
    var g2base :int2d = g1base 
    if (y + step) < size_y then 
      g2base = {x0, y0 + y + step}
    end

    for x = 0, size_x, step do
      r_image[base + {x, 0}].value += (r_image[g1base + {x,0}].value + r_image[g2base + {x,0}].value) / 4
    end 

  end

  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Lift y took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
end


task saveImage(r_image : region(ispace(int2d), Pixel),
                filename : rawstring)
where
  reads(r_image.value)
do
  png.write_png_file(filename,
                     __physical(r_image.value),
                     __fields(r_image.value),
                     r_image.bounds)
end

task toplevel()
  var config : WaveletConfig
  config:initialize_from_command()
  
  var size_image = png.get_image_size(config.filename_image)
  var r_image = region(ispace(int2d, size_image), Pixel) 
  initialize(r_image, config.filename_image) 
  
  var step : uint32 = 1
  
  -- Once we divide r_image into colors, can we still get size of each color
  -- partition? Also we want to divide it such that its a bunch of rows.
  -- Will we need the exact same partition while decompressing?
  
  var size_x : uint32 = r_image.bounds.hi.x
  var size_y : uint32 = r_image.bounds.hi.y
  
  -- Want to launch the while loop on each region separately.
  while (step < size_x or step < size_y) do
    if (step < size_x) then
      liftX(r_image, step)
    end
    if (step < size_y) then
      liftY(r_image, step)
    end
    step = step*2
  end

  -- This shouldn't start running until liftX is finished because of the
  -- dependency. 
  saveImage(r_image, config.filename_edge)
end

regentlib.start(toplevel)
